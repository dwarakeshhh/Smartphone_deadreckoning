package com.deadreckoning.tracker

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager

/**
 * Reads accelerometer, gyroscope and magnetometer.
 *
 * A packet is emitted on every accelerometer event (rate-limited to [rateHz]) and carries the
 * most recent gyroscope and magnetometer readings. The timestamp is the accelerometer event
 * time in ms (SensorEvent.timestamp is nanoseconds on the monotonic elapsed-realtime clock).
 *
 * Android device axes (portrait): X right, Y up along the screen, Z out of the screen.
 */
class SensorHub(context: Context, private val onPacket: (SensorPacket) -> Unit) : SensorEventListener {

    private val sm = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    val accel: Sensor? = sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    val gyro: Sensor? = sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    val mag: Sensor? = sm.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)

    @Volatile
    var latestAcc = FloatArray(3)
        private set
    @Volatile
    var latestGyro = FloatArray(3)
        private set
    @Volatile
    var latestMag = FloatArray(3)
        private set
    @Volatile var emitting = false
    @Volatile var rateHz = 50

    private var lastEmitNs = 0L
    private var registered = false
    var magAvailable = mag != null

    fun describe(): String = buildString {
        fun line(name: String, s: Sensor?) =
            append(name).append(": ").append(if (s == null) "NOT AVAILABLE" else "OK - ${s.name}").append('\n')
        line("Accelerometer", accel)
        line("Gyroscope", gyro)
        line("Magnetometer", mag)
        append("(magnetometer optional: without it heading is relative to the start pose)")
    }

    fun requiredSensorsPresent() = accel != null && gyro != null

    fun start(hz: Int) {
        rateHz = hz.coerceIn(5, 200)
        stop()
        val periodUs = 1_000_000 / rateHz
        // Request a rate at least as fast as wanted; emission is throttled in onSensorChanged.
        val reqUs = (periodUs / 2).coerceAtLeast(5_000)
        accel?.let { sm.registerListener(this, it, reqUs) }
        gyro?.let { sm.registerListener(this, it, reqUs) }
        mag?.let { sm.registerListener(this, it, reqUs.coerceAtLeast(20_000)) }
        registered = true
    }

    fun stop() {
        if (registered) sm.unregisterListener(this)
        registered = false
    }

    override fun onSensorChanged(e: SensorEvent) {
        when (e.sensor.type) {
            Sensor.TYPE_GYROSCOPE -> latestGyro = e.values.copyOf(3)
            Sensor.TYPE_MAGNETIC_FIELD -> latestMag = e.values.copyOf(3)
            Sensor.TYPE_ACCELEROMETER -> {
                val a = e.values.copyOf(3)
                latestAcc = a
                if (emitting) {
                    val periodNs = 1_000_000_000L / rateHz
                    // allow 10 % jitter so a nominal 50 Hz stream is not thinned to 25 Hz
                    if (e.timestamp - lastEmitNs >= periodNs * 9 / 10) {
                        lastEmitNs = e.timestamp
                        onPacket(SensorPacket(e.timestamp / 1e6, a, latestGyro, latestMag))
                    }
                }
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}
