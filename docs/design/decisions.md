# Accepted decisions

## Repository branches

- `main` contains the stable `v0.2.3` application.
- `dev` contains experiments, new measurement modules, and the next
  architecture.
- Focused feature branches may be created from `dev` when useful.
- Stable releases reach `main` only after development has been organized and
  verified.

## Application and modules

- The main application is represented by `SpectrumApplication`.
- Measurement modules are classes derived from a common abstract base class.
- Every module receives the full `SpectrumApplication`, not a restricted module
  context.
- Public application subsystems form the supported module API; underscore-prefixed
  implementation details are not stable API.
- Module `__init__` stores local fields only. Access requiring a fully initialized
  application belongs in lifecycle methods.
- Per-measurement state, module controls, running measurement jobs, and plot
  renderers are separate objects rather than one large module class.

## Dependencies and resources

- Modules do not import or control other measurement modules directly.
- Reusable DSP, measurement math, generators, and neutral result types are
  developed in `audioanalysis` and consumed by the application modules.
- `audioanalysis` must remain independent of Dear PyGui, `SpectrumApplication`,
  project files, and module-specific UI state.
- Application code must not duplicate algorithms already exposed by
  `audioanalysis`.
- Physical audio I/O belongs to the application core and is exposed through an
  abstract application-level interface.
- DPG calls are confined to UI-facing application and view code; background
  measurement code returns data rather than manipulating widgets.

## Module loading

- Initial development uses built-in modules with a common lifecycle.
- A module manager will later discover modules from the modules directory.
- Hot reload is not required for the first implementation.

## Open decisions

- Plot workspace, layer compatibility, and axis allocation API.
- Project schema and calibration-storage boundaries.
- Concrete signal/slot implementation and worker-to-UI synchronization.
- Exact module discovery manifest and failure-isolation behavior.
