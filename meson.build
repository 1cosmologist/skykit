project(
  'skykit',
  version: files('VERSION'),
  meson_version: '>= 0.64.0',
  default_options: [
    'warning_level=2',
  ],
)

py = import('python').find_installation(pure: true)

subdir('skykit')

test('python-unittests', py,
  args: ['-m', 'unittest', 'discover', '-s', 'tests', '-v'],
  workdir: meson.project_source_root()
)

