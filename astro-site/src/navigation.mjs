import apiSidebar from './data/api-sidebar.json' with { type: 'json' };

const page = (label, slug) => ({ label, slug });
const group = (label, items) => ({ label, items, collapsed: true });
const apiNames = { ecg: 'ECG', ppg: 'PPG', rsp: 'Respiration', imu: 'IMU', hrv: 'HRV', signal: 'Signal utilities' };
const apiGroups = Object.entries(Object.groupBy(apiSidebar, (item) => item.label.split('.')[1] || 'Package'))
  .map(([key, items]) => group(apiNames[key] || key, items));

export const sections = [
  { label: 'Home', href: '/physiokit/', sidebar: false },
  {
    label: 'Getting started',
    href: '/physiokit/tutorial/quickstart/',
    sidebar: [page('Install and quickstart', 'tutorial/quickstart')],
  },
  {
    label: 'Signals and examples',
    href: '/physiokit/reference/',
    sidebar: [
      page('Overview', 'reference'),
      page('ECG', 'reference/ecg'),
      page('PPG', 'reference/ppg'),
      page('Respiration', 'reference/rsp'),
      page('IMU', 'reference/imu'),
      page('Heart rate variability', 'reference/hrv'),
      page('Signal utilities', 'reference/signal'),
    ],
  },
  {
    label: 'Python API',
    href: '/physiokit/api/',
    sidebar: [page('API catalog', 'api'), ...apiGroups],
  },
];
