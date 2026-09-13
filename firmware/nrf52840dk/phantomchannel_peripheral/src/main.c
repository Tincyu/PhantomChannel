/*
 * Controlled PhantomChannel BLE peripheral.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <zephyr/bluetooth/addr.h>
#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/conn.h>
#include <zephyr/bluetooth/gatt.h>
#include <zephyr/bluetooth/hci.h>
#include <zephyr/bluetooth/uuid.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/byteorder.h>
#include <zephyr/sys/printk.h>

#define PHANTOM_SERVICE_UUID_VAL \
	BT_UUID_128_ENCODE(0x70630001, 0x5048, 0x414e, 0x544f, 0x4d4348414e4c)
#define PHANTOM_DATA_UUID_VAL \
	BT_UUID_128_ENCODE(0x70630002, 0x5048, 0x414e, 0x544f, 0x4d4348414e4c)

#define PHANTOM_FRAME_OVERHEAD 6U
#define PHANTOM_MARKER_LEN 3U
#define PHANTOM_MAX_COVERT_LEN 240U
#define PHANTOM_NORMAL_PREFIX_LEN CONFIG_PHANTOMCHANNEL_NORMAL_PREFIX_LEN
#define PHANTOM_COVERT_LEN CONFIG_PHANTOMCHANNEL_COVERT_LEN
#define PHANTOM_COVERT_MARKER_BYTE ((uint8_t)CONFIG_PHANTOMCHANNEL_COVERT_MARKER_BYTE)
/*
 * Keep RTT focused on PHANTOM_LL_TX records during long IQ captures.  The
 * controller-side record carries the seq/access-address/channel tuple needed
 * by the SDR matcher; emitting a full application hex dump for every 20 ms
 * notification can overflow the RTT ring and lose those records.
 */
#define PHANTOM_VERBOSE_LOG_MAX_COVERT_LEN 0U
#define PHANTOM_FRAME_LEN (PHANTOM_FRAME_OVERHEAD + PHANTOM_COVERT_LEN)
#define PHANTOM_NOTIFY_LEN (PHANTOM_NORMAL_PREFIX_LEN + PHANTOM_MARKER_LEN + PHANTOM_FRAME_LEN)

BUILD_ASSERT(PHANTOM_COVERT_LEN >= 1U);
BUILD_ASSERT(PHANTOM_COVERT_LEN <= PHANTOM_MAX_COVERT_LEN);
BUILD_ASSERT(PHANTOM_NOTIFY_LEN <= 248);

static struct bt_uuid_128 phantom_service_uuid = BT_UUID_INIT_128(PHANTOM_SERVICE_UUID_VAL);
static struct bt_uuid_128 phantom_data_uuid = BT_UUID_INIT_128(PHANTOM_DATA_UUID_VAL);

static const struct bt_data ad[] = {
	BT_DATA_BYTES(BT_DATA_FLAGS, BT_LE_AD_GENERAL | BT_LE_AD_NO_BREDR),
	BT_DATA_BYTES(BT_DATA_UUID128_ALL, PHANTOM_SERVICE_UUID_VAL),
};

static const struct bt_data sd[] = {
	BT_DATA(BT_DATA_NAME_COMPLETE, CONFIG_BT_DEVICE_NAME,
		sizeof(CONFIG_BT_DEVICE_NAME) - 1),
};

static struct bt_conn *current_conn;
static bool notify_enabled;
static uint16_t phantom_seq;
static uint8_t identity_id = BT_ID_DEFAULT;
static struct k_work_delayable notify_work;
static struct k_work_delayable advertising_work;
K_MUTEX_DEFINE(conn_mutex);

static uint8_t notify_value[PHANTOM_NOTIFY_LEN];
static uint8_t covert_payload[PHANTOM_COVERT_LEN];
static uint8_t covert_frame[PHANTOM_FRAME_LEN];
static char covert_hex_buf[(PHANTOM_MAX_COVERT_LEN * 2U) + 1U];
static char covert_data_hex_buf[((PHANTOM_MAX_COVERT_LEN - 1U) * 2U) + 1U];
static char frame_hex_buf[((PHANTOM_FRAME_OVERHEAD + PHANTOM_MAX_COVERT_LEN) * 2U) + 1U];

static void ccc_changed(const struct bt_gatt_attr *attr, uint16_t value);

BT_GATT_SERVICE_DEFINE(phantom_service,
	BT_GATT_PRIMARY_SERVICE(&phantom_service_uuid),
	BT_GATT_CHARACTERISTIC(&phantom_data_uuid.uuid, BT_GATT_CHRC_NOTIFY,
			       BT_GATT_PERM_NONE, NULL, NULL, NULL),
	BT_GATT_CCC(ccc_changed, BT_GATT_PERM_READ | BT_GATT_PERM_WRITE),
);

static void hex_encode(const uint8_t *data, size_t len, char *out, size_t out_len)
{
	static const char digits[] = "0123456789abcdef";
	size_t pos = 0U;

	if (out_len == 0U) {
		return;
	}

	for (size_t i = 0; i < len && (pos + 2U) < out_len; i++) {
		out[pos++] = digits[data[i] >> 4];
		out[pos++] = digits[data[i] & 0x0f];
	}
	out[pos] = '\0';
}

static void fill_covert_payload(uint16_t seq)
{
	ARG_UNUSED(seq);

	covert_payload[0] = PHANTOM_COVERT_MARKER_BYTE;
	for (size_t i = 1; i < sizeof(covert_payload); i++) {
		covert_payload[i] = (uint8_t)(((i - 1U) % 255U) + 1U);
	}
}

static void build_covert_frame(uint16_t seq)
{
	uint8_t check = 0U;

	covert_frame[0] = 0x50;
	covert_frame[1] = 0x43;
	sys_put_le16(seq, &covert_frame[2]);
	covert_frame[4] = (uint8_t)sizeof(covert_payload);
	memcpy(&covert_frame[5], covert_payload, sizeof(covert_payload));

	for (size_t i = 0; i < sizeof(covert_frame) - 1U; i++) {
		check ^= covert_frame[i];
	}
	covert_frame[sizeof(covert_frame) - 1U] = check;
}

static void build_notify_value(void)
{
	memset(notify_value, 0, sizeof(notify_value));
	notify_value[0] = 0x06;
	for (size_t i = 1; i < PHANTOM_NORMAL_PREFIX_LEN; i++) {
		notify_value[i] = (uint8_t)i;
	}
	notify_value[PHANTOM_NORMAL_PREFIX_LEN + 0U] = 0xaa;
	notify_value[PHANTOM_NORMAL_PREFIX_LEN + 1U] = 0xaa;
	notify_value[PHANTOM_NORMAL_PREFIX_LEN + 2U] = 0x00;
	memcpy(&notify_value[PHANTOM_NORMAL_PREFIX_LEN + PHANTOM_MARKER_LEN],
	       covert_frame, sizeof(covert_frame));
}

static void log_phantom_tx(uint16_t seq, int err)
{
	if (sizeof(covert_payload) > PHANTOM_VERBOSE_LOG_MAX_COVERT_LEN) {
		if (err == 0 && seq >= 8U && (seq % 16U) != 0U) {
			return;
		}

		printk("PHANTOM_TX {\"session_id\":0,\"seq\":%u,\"conn_event\":null,"
		       "\"channel\":null,\"phy\":\"1M\",\"normal_pdu_len\":%u,"
		       "\"covert_len\":%u,\"covert_marker_hex\":\"%02x\","
		       "\"covert_data_len\":%u,"
		       "\"covert_pattern\":\"marker_fixed_01_to_ff\","
		       "\"frame_pattern\":\"pc_v1_xor\","
		       "\"timestamp_us\":%lld,\"status\":%d}\n",
		       seq, (unsigned int)PHANTOM_NORMAL_PREFIX_LEN,
		       (unsigned int)sizeof(covert_payload),
		       (unsigned int)PHANTOM_COVERT_MARKER_BYTE,
		       (unsigned int)(sizeof(covert_payload) - 1U),
		       k_uptime_get() * 1000LL, err);
		return;
	}

	hex_encode(covert_payload, sizeof(covert_payload), covert_hex_buf, sizeof(covert_hex_buf));
	hex_encode(&covert_payload[1], sizeof(covert_payload) - 1U,
		   covert_data_hex_buf, sizeof(covert_data_hex_buf));
	hex_encode(covert_frame, sizeof(covert_frame), frame_hex_buf, sizeof(frame_hex_buf));

	printk("PHANTOM_TX {\"session_id\":0,\"seq\":%u,\"conn_event\":null,"
	       "\"channel\":null,\"phy\":\"1M\",\"normal_pdu_len\":%u,"
	       "\"covert_len\":%u,\"covert_marker_hex\":\"%02x\","
	       "\"covert_hex\":\"%s\",\"covert_data_len\":%u,"
	       "\"covert_data_hex\":\"%s\",\"frame_hex\":\"%s\","
	       "\"timestamp_us\":%lld,\"status\":%d}\n",
	       seq, (unsigned int)PHANTOM_NORMAL_PREFIX_LEN,
	       (unsigned int)sizeof(covert_payload),
	       (unsigned int)PHANTOM_COVERT_MARKER_BYTE,
	       covert_hex_buf, (unsigned int)(sizeof(covert_payload) - 1U),
	       covert_data_hex_buf, frame_hex_buf,
	       k_uptime_get() * 1000LL, err);
}

static void schedule_notification(void)
{
	k_work_reschedule(&notify_work, K_MSEC(CONFIG_PHANTOMCHANNEL_NOTIFY_INTERVAL_MS));
}

static void notify_handler(struct k_work *work)
{
	struct bt_conn *conn = NULL;
	uint16_t seq;
	int err;

	ARG_UNUSED(work);

	k_mutex_lock(&conn_mutex, K_FOREVER);
	if (current_conn != NULL) {
		conn = bt_conn_ref(current_conn);
	}
	k_mutex_unlock(&conn_mutex);

	if (conn == NULL || !notify_enabled) {
		if (conn != NULL) {
			bt_conn_unref(conn);
		}
		return;
	}

	seq = phantom_seq++;
	fill_covert_payload(seq);
	build_covert_frame(seq);
	build_notify_value();

	err = bt_gatt_notify(conn, &phantom_service.attrs[2], notify_value,
			     sizeof(notify_value));
	log_phantom_tx(seq, err);

	bt_conn_unref(conn);
	schedule_notification();
}

static void ccc_changed(const struct bt_gatt_attr *attr, uint16_t value)
{
	ARG_UNUSED(attr);

	notify_enabled = (value == BT_GATT_CCC_NOTIFY);
	printk("PHANTOM_CCC {\"notify\":%u,\"timestamp_us\":%lld}\n",
	       notify_enabled ? 1U : 0U, k_uptime_get() * 1000LL);
	if (notify_enabled) {
		schedule_notification();
	}
}

static int configure_static_identity(void)
{
	bt_addr_le_t identity_addr;
	char addr_str[BT_ADDR_LE_STR_LEN];
	int id;
	int err;

	err = bt_addr_le_from_str(CONFIG_PHANTOMCHANNEL_STATIC_ADDRESS, "random",
				  &identity_addr);
	if (err != 0) {
		printk("PHANTOM_IDENTITY_ERROR {\"address\":\"%s\",\"parse_error\":%d}\n",
		       CONFIG_PHANTOMCHANNEL_STATIC_ADDRESS, err);
		return err;
	}

	if (!BT_ADDR_IS_STATIC(&identity_addr.a)) {
		printk("PHANTOM_IDENTITY_ERROR {\"address\":\"%s\",\"error\":\"not_static_random\"}\n",
		       CONFIG_PHANTOMCHANNEL_STATIC_ADDRESS);
		return -EINVAL;
	}

	id = bt_id_create(&identity_addr, NULL);
	if (id < 0) {
		printk("PHANTOM_IDENTITY_ERROR {\"address\":\"%s\",\"id\":%d}\n",
		       CONFIG_PHANTOMCHANNEL_STATIC_ADDRESS, id);
		return id;
	}

	identity_id = (uint8_t)id;
	bt_addr_le_to_str(&identity_addr, addr_str, sizeof(addr_str));
	printk("PHANTOM_IDENTITY {\"id\":%d,\"address\":\"%s\"}\n", id, addr_str);
	return 0;
}

static int configure_low_band_channel_map(void)
{
	if (!IS_ENABLED(CONFIG_PHANTOMCHANNEL_LOW_BAND_CHAN_MAP)) {
		printk("PHANTOM_CHAN_MAP {\"enabled\":0,\"reason\":\"disabled\"}\n");
		return 0;
	}

	/* BLE data channels 0-16 map to 2404-2438 MHz, within a 2420 MHz
	 * center / 40 MHz SDR capture. Advertising channel 37 is 2402 MHz and
	 * is outside the connection data channel map.
	 */
	uint8_t chan_map[5] = { 0xff, 0xff, 0x01, 0x00, 0x00 };
	int err = bt_le_set_chan_map(chan_map);

	printk("PHANTOM_CHAN_MAP {\"enabled\":1,\"status\":%d,"
	       "\"map_hex\":\"ffff010000\",\"data_channels\":\"0-16\","
	       "\"freq_mhz\":\"2404-2438\"}\n", err);
	return err;
}

static int start_advertising(void)
{
	struct bt_le_adv_param adv_param = *BT_LE_ADV_CONN_FAST_1;
	int err;

	adv_param.id = identity_id;
	err = bt_le_adv_start(&adv_param, ad, ARRAY_SIZE(ad), sd, ARRAY_SIZE(sd));
	printk("PHANTOM_ADV_START {\"status\":%d,\"timestamp_us\":%lld}\n",
	       err, k_uptime_get() * 1000LL);
	return err;
}

static void advertising_handler(struct k_work *work)
{
	int err;

	ARG_UNUSED(work);

	err = start_advertising();
	if (err == -ENOMEM || err == -EAGAIN) {
		k_work_reschedule(&advertising_work, K_MSEC(100));
	}
}

static void connected(struct bt_conn *conn, uint8_t err)
{
	struct bt_le_conn_param fast_param =
		BT_LE_CONN_PARAM_INIT(BT_GAP_US_TO_CONN_INTERVAL(7500),
				      BT_GAP_US_TO_CONN_INTERVAL(7500),
				      0,
				      BT_GAP_MS_TO_CONN_TIMEOUT(4000));
	char addr[BT_ADDR_LE_STR_LEN];
	int param_err;

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));
	if (err != 0U) {
		printk("PHANTOM_CONNECT_FAILED {\"peer\":\"%s\",\"error\":%u,\"timestamp_us\":%lld}\n",
		       addr, err, k_uptime_get() * 1000LL);
		return;
	}

	k_mutex_lock(&conn_mutex, K_FOREVER);
	if (current_conn != NULL) {
		bt_conn_unref(current_conn);
	}
	current_conn = bt_conn_ref(conn);
	k_mutex_unlock(&conn_mutex);

	phantom_seq = 0U;
	printk("PHANTOM_CONNECTED {\"peer\":\"%s\",\"timestamp_us\":%lld}\n",
	       addr, k_uptime_get() * 1000LL);

	param_err = bt_conn_le_param_update(conn, &fast_param);
	printk("PHANTOM_CONN_PARAM_UPDATE {\"interval_min_us\":7500,"
	       "\"interval_max_us\":7500,\"latency\":0,\"timeout_ms\":4000,"
	       "\"status\":%d,\"timestamp_us\":%lld}\n",
	       param_err, k_uptime_get() * 1000LL);
}

static void disconnected(struct bt_conn *conn, uint8_t reason)
{
	char addr[BT_ADDR_LE_STR_LEN];

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));
	printk("PHANTOM_DISCONNECTED {\"peer\":\"%s\",\"reason\":%u,\"timestamp_us\":%lld}\n",
	       addr, reason, k_uptime_get() * 1000LL);

	notify_enabled = false;
	k_work_cancel_delayable(&notify_work);

	k_mutex_lock(&conn_mutex, K_FOREVER);
	if (current_conn != NULL) {
		bt_conn_unref(current_conn);
		current_conn = NULL;
	}
	k_mutex_unlock(&conn_mutex);

	k_work_reschedule(&advertising_work, K_MSEC(100));
}

BT_CONN_CB_DEFINE(conn_callbacks) = {
	.connected = connected,
	.disconnected = disconnected,
};

int main(void)
{
	int err;

	k_work_init_delayable(&notify_work, notify_handler);
	k_work_init_delayable(&advertising_work, advertising_handler);

	err = bt_enable(NULL);
	printk("PHANTOM_BOOT {\"bt_enable\":%d,\"device\":\"%s\",\"covert_len\":%u,"
	       "\"covert_marker_hex\":\"%02x\",\"covert_data_len\":%u,"
	       "\"notify_interval_ms\":%u,\"low_band_chan_map\":%u,"
	       "\"chan_sel_2\":%u,\"chan_sel_algorithm\":\"%s\"}\n",
	       err, CONFIG_BT_DEVICE_NAME, (unsigned int)PHANTOM_COVERT_LEN,
	       (unsigned int)PHANTOM_COVERT_MARKER_BYTE,
	       (unsigned int)(PHANTOM_COVERT_LEN - 1U),
	       (unsigned int)CONFIG_PHANTOMCHANNEL_NOTIFY_INTERVAL_MS,
	       IS_ENABLED(CONFIG_PHANTOMCHANNEL_LOW_BAND_CHAN_MAP) ? 1U : 0U,
	       IS_ENABLED(CONFIG_BT_CTLR_CHAN_SEL_2) ? 1U : 0U,
	       IS_ENABLED(CONFIG_BT_CTLR_CHAN_SEL_2) ? "CSA#2" : "CSA#1");
	if (err != 0) {
		return 0;
	}

	(void)configure_low_band_channel_map();

	err = configure_static_identity();
	if (err != 0) {
		return 0;
	}

	(void)start_advertising();

	while (true) {
		k_sleep(K_SECONDS(1));
	}

	return 0;
}
