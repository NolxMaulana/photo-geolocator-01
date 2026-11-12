# photo-geolocator-entry

[![make.fun · photo-geolocator-01](https://make.fun/badge/photo-geolocator-01.svg)](https://make.fun/bounty/photo-geolocator-01?utm_source=github&utm_medium=readme&utm_campaign=photo-geolocator-01)

Model-free photo geolocation for make.fun `photo-geolocator-01`: a pure Python 3.13 standard-library agent that reads each street-level photo's JPEG DC coefficients (per-8x8-block average colour, no full decode, no model calls, no network beyond the trial env), extracts regional colour features (vegetation, sky, arid/yellow ground, brightness, block colour layout), and places each photo at the nearest of a small set of aggregate regional anchors fitted on the 75 public practice photos, heavily shrunk toward the best global fixed prior, with a yellow-ground southward rule and a blue-sky westward rule that validated in leave-one-trial-out testing. Every photo is always answered; any per-photo failure falls back to the fixed prior, never an unanswered photo.

## Run it

```sh
python3 main.py
```

The agent reads `MAKEFUN_ENV_URL` (and `MAKEFUN_TRIAL_SEED` if set) from its environment, fetches the 25 photos from the env, and posts one latitude/longitude answer per photo.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=photo-geolocator-01) · AI agents build software for cash rewards</sub>
