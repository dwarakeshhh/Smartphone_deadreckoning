# Android app - IMU Tracker (Kotlin)

Streams accelerometer + gyroscope + magnetometer to the laptop over a WebSocket.

## NOT BUILT IN THE DELIVERY ENVIRONMENT
This project was generated in an environment without the Android SDK, so the app has **not been compiled
or run on a device yet**. The Python side was fully tested against a protocol-identical simulator; the first
build in Android Studio is the first real test of this app. If Gradle or the compiler reports anything,
fix it in the file named in the message (the code is small: 4 Kotlin files).

## Build & install
1. Install **Android Studio** (Koala or newer) - it brings the SDK and JDK 17.
2. `File > Open` -> choose this `android_app/` folder. Let Gradle sync (needs internet once to download dependencies).
   If asked to create/upgrade the Gradle wrapper, accept.
3. Enable **Developer options + USB debugging** on the phone, connect it, press **Run ▶**.
   (Or `Build > Build APK(s)` and install the APK.)

## Use
1. Phone and laptop on the **same Wi-Fi**. Start the Python server on the laptop; it prints `ws://<ip>:8000/ws`.
2. Enter the laptop IP (and port 8000). Sampling rate default 50 Hz. Press **CONNECT** -> green "CONNECTED".
3. Put the phone **flat and still** at the start point, press **START TRACKING**, wait ~1.5 s (calibration).
4. Move the phone. Press **STOP TRACKING** to see the summary on the laptop.
- **RESET** asks the server to forget the session and re-calibrate.
- If Wi-Fi drops, the app reconnects automatically (1 s -> 5 s back-off) and resumes the session.
- The screen stays on while the app is open (`FLAG_KEEP_SCREEN_ON`) and a partial wake lock is held while tracking.

## Files
| file | purpose |
|---|---|
| `MainActivity.kt` | UI, buttons, connection/tracking state |
| `SensorHub.kt` | SensorManager listeners, rate limiting, packet building |
| `TrackerSocket.kt` | OkHttp WebSocket client + auto-reconnect |
| `SensorPacket.kt` | JSON packet (protocol in `docs/protocol.md`) |

## Troubleshooting
* **Cannot connect**: wrong IP, different Wi-Fi (guest/AP-isolation networks block device-to-device traffic), or Windows Firewall blocks port 8000 (see main README).
* **Cleartext not permitted**: the manifest already sets `usesCleartextTraffic="true"` for `ws://`.
* **Gyroscope missing**: tracking is refused (needed for orientation). Magnetometer missing: works, heading becomes relative.
* Packets sent not rising: you must press START TRACKING (live values show without it, but nothing is streamed).
