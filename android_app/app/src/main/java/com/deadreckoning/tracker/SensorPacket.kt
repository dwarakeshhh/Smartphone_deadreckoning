package com.deadreckoning.tracker

import org.json.JSONObject

/**
 * One fused sensor sample. Units (Android native):
 *  acc  m/s^2 (includes gravity), gyro rad/s, mag microtesla,
 *  timestamp = milliseconds on the phone's monotonic sensor clock (elapsed-realtime based).
 */
data class SensorPacket(
    val timestampMs: Double,
    val acc: FloatArray,
    val gyro: FloatArray,
    val mag: FloatArray
) {
    fun toJson(): String {
        val o = JSONObject()
        o.put("timestamp", timestampMs)
        o.put("acc_x", acc[0].toDouble()); o.put("acc_y", acc[1].toDouble()); o.put("acc_z", acc[2].toDouble())
        o.put("gyro_x", gyro[0].toDouble()); o.put("gyro_y", gyro[1].toDouble()); o.put("gyro_z", gyro[2].toDouble())
        o.put("mag_x", mag[0].toDouble()); o.put("mag_y", mag[1].toDouble()); o.put("mag_z", mag[2].toDouble())
        o.put("source", "android")
        return o.toString()
    }
}
