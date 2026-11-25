package com.deadreckoning.tracker

import android.content.Context
import android.graphics.Color
import android.os.Bundle
import android.os.PowerManager
import android.view.WindowManager
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.os.Handler
import android.os.Looper
import androidx.appcompat.app.AppCompatActivity
import org.json.JSONObject

class MainActivity : AppCompatActivity() {

    private lateinit var tvConn: TextView
    private lateinit var tvTracking: TextView
    private lateinit var tvServer: TextView
    private lateinit var tvSensors: TextView
    private lateinit var tvLive: TextView
    private lateinit var etIp: EditText
    private lateinit var etPort: EditText
    private lateinit var etRate: EditText
    private lateinit var btnConnect: Button
    private lateinit var btnStart: Button
    private lateinit var btnStop: Button
    private lateinit var btnReset: Button

    private lateinit var hub: SensorHub
    private lateinit var socket: TrackerSocket
    private var wakeLock: PowerManager.WakeLock? = null

    private var connState = ConnState.DISCONNECTED
    private var tracking = false          // user pressed START (kept across reconnects)
    private val ui = Handler(Looper.getMainLooper())
    private val uiTick = object : Runnable {
        override fun run() { refreshLive(); ui.postDelayed(this, 100) }   // 10 Hz UI refresh
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)   // keep collecting while the screen is on

        tvConn = findViewById(R.id.tvConnStatus); tvTracking = findViewById(R.id.tvTracking)
        tvServer = findViewById(R.id.tvServer); tvSensors = findViewById(R.id.tvSensors)
        tvLive = findViewById(R.id.tvLive)
        etIp = findViewById(R.id.etIp); etPort = findViewById(R.id.etPort); etRate = findViewById(R.id.etRate)
        btnConnect = findViewById(R.id.btnConnect); btnStart = findViewById(R.id.btnStart)
        btnStop = findViewById(R.id.btnStop); btnReset = findViewById(R.id.btnReset)

        val prefs = getSharedPreferences("tracker", Context.MODE_PRIVATE)
        etIp.setText(prefs.getString("ip", ""))
        etPort.setText(prefs.getString("port", "8000"))
        etRate.setText(prefs.getString("rate", "50"))

        hub = SensorHub(this) { packet ->
            // called on the sensor thread; OkHttp's send() is thread-safe
            socket.sendText(packet.toJson())
        }
        socket = TrackerSocket(
            onState = { s, detail -> connState = s; tvServer.text = "Server: $detail"; renderState() },
            onServerMessage = { handleServerMessage(it) },
            onConnected = { if (tracking) socket.sendControl("start") }   // resume after a reconnect
        )
        tvSensors.text = hub.describe()

        btnConnect.setOnClickListener {
            if (connState == ConnState.DISCONNECTED) {
                val ip = etIp.text.toString().trim()
                val port = etPort.text.toString().trim().ifEmpty { "8000" }
                if (ip.isEmpty()) { tvServer.text = "Server: enter the laptop IP address first"; return@setOnClickListener }
                prefs.edit().putString("ip", ip).putString("port", port)
                    .putString("rate", etRate.text.toString().trim()).apply()
                socket.connect("ws://$ip:$port/ws")
            } else {
                stopTracking(sendStop = true)
                socket.disconnect()
            }
        }
        btnStart.setOnClickListener { startTracking() }
        btnStop.setOnClickListener { stopTracking(sendStop = true) }
        btnReset.setOnClickListener {
            socket.sendControl("reset")
            tvServer.text = "Server: reset requested"
        }
        renderState()
    }

    override fun onResume() {
        super.onResume()
        // sensors are registered while the app is visible so live values show even before tracking
        if (hub.requiredSensorsPresent()) hub.start(currentRate())
        ui.post(uiTick)
        hub.emitting = tracking
    }

    override fun onPause() {
        super.onPause()
        ui.removeCallbacks(uiTick)
        // Keep streaming only if tracking is active (screen-on use case); otherwise save battery.
        if (!tracking) hub.stop()
    }

    override fun onDestroy() {
        stopTracking(sendStop = false)
        socket.disconnect()
        hub.stop()
        super.onDestroy()
    }

    private fun currentRate(): Int = etRate.text.toString().trim().toIntOrNull()?.coerceIn(5, 200) ?: 50

    private fun startTracking() {
        if (!hub.requiredSensorsPresent()) { tvServer.text = "Server: accelerometer/gyroscope missing - cannot track"; return }
        if (!socket.isOpen) { tvServer.text = "Server: press CONNECT first"; return }
        hub.start(currentRate())
        tracking = true
        socket.sendControl("start")
        hub.emitting = true
        acquireWakeLock()
        renderState()
    }

    private fun stopTracking(sendStop: Boolean) {
        if (!tracking && !sendStop) return
        tracking = false
        hub.emitting = false
        if (sendStop) socket.sendControl("stop")
        releaseWakeLock()
        renderState()
    }

    private fun acquireWakeLock() {
        val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
        if (wakeLock == null) wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "tracker:sensors")
        if (wakeLock?.isHeld == false) wakeLock?.acquire(60 * 60 * 1000L)
    }

    private fun releaseWakeLock() { if (wakeLock?.isHeld == true) wakeLock?.release() }

    private fun handleServerMessage(text: String) {
        try {
            val o = JSONObject(text)
            when (o.optString("type")) {
                "status" -> tvServer.text = "Server: ${o.optString("ui_status")} · phase ${o.optString("phase")} · " +
                        "received ${o.optInt("packets")} · state ${o.optString("state")}"
                "error" -> tvServer.text = "Server rejected data: ${o.optString("message")}"
                "hello" -> tvServer.text = "Server: hello (expects ~${o.optDouble("expected_rate_hz", 50.0).toInt()} Hz)"
            }
        } catch (_: Exception) { /* ignore malformed server text */ }
    }

    private fun renderState() {
        val (txt, color) = when (connState) {
            ConnState.CONNECTED -> (if (tracking) "CONNECTED · TRACKING" else "CONNECTED") to R.color.ok
            ConnState.CONNECTING -> "CONNECTING..." to R.color.warn
            ConnState.RECONNECTING -> "RECONNECTING..." to R.color.warn
            ConnState.DISCONNECTED -> "DISCONNECTED" to R.color.idle
        }
        tvConn.text = txt
        tvConn.setBackgroundColor(getColor(color))
        tvConn.setTextColor(Color.WHITE)
        btnConnect.text = if (connState == ConnState.DISCONNECTED) "CONNECT" else "DISCONNECT"
        btnStart.isEnabled = connState == ConnState.CONNECTED && !tracking
        btnStop.isEnabled = tracking
        btnReset.isEnabled = connState == ConnState.CONNECTED
    }

    private fun refreshLive() {
        val a = hub.latestAcc; val g = hub.latestGyro; val m = hub.latestMag
        tvLive.text = String.format(
            "ACC  x %7.3f  y %7.3f  z %7.3f m/s²\nGYRO x %7.3f  y %7.3f  z %7.3f rad/s\nMAG  x %7.2f  y %7.2f  z %7.2f µT",
            a[0], a[1], a[2], g[0], g[1], g[2], m[0], m[1], m[2])
        tvTracking.text = "Tracking: ${if (tracking) "ON" else "OFF"}   ·   Packets sent: ${socket.packetsSent.get()}" +
                (if (socket.packetsDropped.get() > 0) "   ·   dropped: ${socket.packetsDropped.get()}" else "")
    }
}
