+ Build macOS and Linux versions and test functionality on those platforms.
- Expand documentation to include setup instructions for different operating systems.

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
- Fix pink noise generator clicking
+ Saving input/output settings on app restart
+ Buffer size selection in settings
+ Update timeplot on record import

v0.2.4
- Get Apple Developer credentials for macOS Developer ID signing and notarization
- Fix missing microphone permission prompt on macOS Tahoe arm64 builds
- Fix manual zoom/pan for the impedance Phase/Dir secondary Y-axis; automatic scaling already works
- Verify SPICE model fitting against real impedance measurements

v0.3
- Implement multi-record mode with averaging
+ Implement semi-analog swept THD+N calculation and display - done

v0.3.1
- Add schema versions and migrations for module state
- Isolate module discovery and initialization failures from the rest of the application
- Track and remove module-owned menu items and callbacks during shutdown
- Implement a Module Manager for loading external modules
- Expand the plot workspace to support multiple plot panels
- Replace the shared frame-callback list with a typed signal/callback API
- Add project-format versioning and migrations to reduce coupling to pickled classes

v0.4
- Implement real-time analyzer mode
- Implement impedance measurement mode
