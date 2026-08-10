+ Build macOS and Linux versions and test functionality on those platforms.

v0.2
+ Implement graph saving (PNG/JPG) - done
+ Implement audio recording exporting/importing to common formats (WAV, MP3). - done
+ Implement last record analysis parameters live update - done
+ Implement project settings saving/loading (with all graphs, records, and settings) - done

v0.2.1
+ Saving analyzer and filtering parameters in record, update fields on changing current record
+ Implement saving and loading projects with all records, graphs, and settings

v0.2.2
+ Fix exception on device disconnection

v0.2.3
+ Fix pink noise generator clicking
+ Saving input/output settings on app restart
+ Buffer size selection in settings
+ Update timeplot on record import

v0.3
+ Implement semi-analog swept THD+N calculation and display - done
+ Implement real-time analyzer mode
+ Implement impedance measurement mode

v0.4
- Automatically choose the RTA smoothing width from point count and band spacing
- Improve the delay-fitting algorithm for phase measurements
- Implement multi-record mode with averaging

v0.5
- Verify SPICE model fitting against real impedance measurements

v0.9
- Implement a Module Manager for manual loading/unloading external modules
- Add schema versions and migrations for module state
- Replace the shared frame-callback list with a typed signal/callback API
- Add project-format versioning and migrations to reduce coupling to pickled classes
- Isolate module discovery and initialization failures from the rest of the application

v1.0
- Get Apple Developer credentials for macOS Developer ID signing and notarization
- Fix missing microphone permission prompt on macOS Tahoe arm64 builds
- Expand documentation