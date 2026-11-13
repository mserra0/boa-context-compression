# Vendored files

The page loads nothing from a CDN. Everything it needs at run time is copied
into the repository and served as written.

## GSAP

| File | Package | Version | Source path in the package |
|---|---|---|---|
| `gsap.min.js` | `gsap` (npm) | 3.15.1 | `dist/gsap.min.js` |
| `ScrollTrigger.min.js` | `gsap` (npm) | 3.15.1 | `dist/ScrollTrigger.min.js` |

Licence: GSAP Standard "No Charge" License, copied from
<https://gsap.com/standard-license> into `GSAP-LICENSE.txt`.

SHA-256:

```
3d1bea374ab1b703cc5c4105358eec63c4465efdc573931dc116e34a0a190c1b  gsap.min.js
b5bb360661f478f0ff6ffa11500d8ce5e2b6aaf6f7288237cdca3aac9548035a  ScrollTrigger.min.js
```

## Fonts (`../fonts/`)

| File | Package | Version | Source path in the package |
|---|---|---|---|
| `source-serif-4-latin-variable.woff2` | `@fontsource-variable/source-serif-4` (npm) | 5.3.1 | `files/source-serif-4-latin-standard-normal.woff2` (axes `wght` 200-900, `opsz` 8-60) |
| `inter-latin-variable.woff2` | `@fontsource-variable/inter` (npm) | 5.3.0 | `files/inter-latin-wght-normal.woff2` (axis `wght` 100-900) |

Both are the Latin subset, which covers the accented letters in the author
names and affiliation. Licence: SIL Open Font License 1.1. The licence texts
next to the fonts (`SourceSerif4-OFL.txt`, `Inter-OFL.txt`) are the upstream
`OFL.txt` files from <https://github.com/google/fonts> (`ofl/sourceserif4/`,
`ofl/inter/`), which carry the correct copyright lines.

SHA-256:

```
f2ea9c12d2fe9bd3a9589b02ad2c0909da88f30938c91adc838c4f4098f9f9e0  source-serif-4-latin-variable.woff2
3100e775e8616cd2611beecfa23a4263d7037586789b43f035236a2e6fbd4c62  inter-latin-variable.woff2
```

## Updating

Run `npm pack <package>@<version>` in a scratch directory, copy the files listed
above, update the versions and hashes here, and check the page in a browser.
