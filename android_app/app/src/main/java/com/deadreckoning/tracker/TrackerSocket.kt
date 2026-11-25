package com.deadreckoning.tracker

import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import android.os.Handler
import android.os.Looper

enum class ConnState { DISCONNECTED, CONNECTING, CONNECTED, RECONNECTING }

/**
 * WebSocket client with automatic reconnection (exponential back-off 1 s -> 5 s).
 * Callbacks are always delivered on the main thread.
 */
class TrackerSocket(
    private val onState: (ConnState, String) -> Unit,
    private val onServerMessage: (String) -> Unit,
    private val onConnected: () -> Unit
) {
    private val client = OkHttpClient.Builder()
        .connectTimeout(4, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.MILLISECONDS)      // no read timeout: server pushes status each second
        .pingInterval(10, TimeUnit.SECONDS)         // detects dead links (e.g. Wi-Fi switched off)
        .build()
    private val main = Handler(Looper.getMainLooper())

    @Volatile private var ws: WebSocket? = null
    @Volatile private var wantConnected = false
    @Volatile private var open = false
    private var url = ""
    private val attempt = AtomicInteger(0)
    private val generation = AtomicInteger(0)

    val packetsSent = AtomicLong(0)
    val packetsDropped = AtomicLong(0)

    val isOpen: Boolean get() = open

    fun connect(url: String) {
        this.url = url
        wantConnected = true
        attempt.set(0)
        openSocket(false)
    }

    fun disconnect() {
        wantConnected = false
        generation.incrementAndGet()
        open = false
        ws?.close(1000, "user disconnect")
        ws = null
        post(ConnState.DISCONNECTED, "Disconnected")
    }

    fun sendText(text: String): Boolean {
        val s = ws ?: return false
        if (!open) return false
        if (s.queueSize() > 200_000) {              // network too slow: drop instead of buffering forever
            packetsDropped.incrementAndGet()
            return false
        }
        val ok = s.send(text)
        if (ok) packetsSent.incrementAndGet() else packetsDropped.incrementAndGet()
        return ok
    }

    /** Control messages are not counted as sensor packets. */
    fun sendControl(command: String): Boolean {
        val s = ws ?: return false
        return open && s.send("""{"type":"control","command":"$command"}""")
    }

    private fun openSocket(isRetry: Boolean) {
        val gen = generation.incrementAndGet()
        post(if (isRetry) ConnState.RECONNECTING else ConnState.CONNECTING, "Connecting to $url ...")
        val request = try {
            Request.Builder().url(url).build()
        } catch (e: IllegalArgumentException) {
            wantConnected = false
            post(ConnState.DISCONNECTED, "Invalid address: ${e.message}")
            return
        }
        ws = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                if (gen != generation.get()) return
                open = true
                attempt.set(0)
                post(ConnState.CONNECTED, "Connected to $url")
                main.post { onConnected() }
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                if (gen != generation.get()) return
                main.post { onServerMessage(text) }
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                if (gen != generation.get()) return
                handleLoss("Closed ($code) $reason")
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                if (gen != generation.get()) return
                handleLoss(t.message ?: "connection failed")
            }
        })
    }

    private fun handleLoss(reason: String) {
        open = false
        if (!wantConnected) return
        val n = attempt.incrementAndGet()
        val delayMs = minOf(5000L, 1000L * (1L shl minOf(n - 1, 3)))
        post(ConnState.RECONNECTING, "Lost: $reason. Retry #$n in ${delayMs / 1000}s")
        main.postDelayed({
            if (wantConnected && !open) openSocket(true)
        }, delayMs)
    }

    private fun post(state: ConnState, detail: String) = main.post { onState(state, detail) }
}
