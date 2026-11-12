# NAPA

NAPA's C core, copied unchanged from [zwatx/NAPA](https://github.com/zwatx/NAPA) (`src/napa_vision/_native`):
branch `feat/recover-shadowed-tags` at commit `4da8f9b`, with NAPA's not yet committed speed and lens work on top
(row-pair labelling, `napa_set_threads`, `napa_set_lens`). It is compiled into `photontargetingJNI` and called
through `NapaDetectorJNI.cpp`. SHA-256 of the copied sources:

```
0d94853298d3cc35bbb3738c2c750eb72e8e144bcb5450935dd5e0afb644dec1  napa_core.c
164e81bac0fd81086e3bcd07f70e488bbb3d66739cbcf0d68bc4d7b674c5a162  napa_core.h
```

NAPA is Copyright (c) 2026 Zachary Wilson and is used here under the GPL-3.0-only; see [NOTICE](NOTICE). The tag
family codewords come from AprilTag under its BSD license; see [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).

To update, copy `napa_core.c`, `napa_core.h`, `families.h`, `families/`, `NOTICE` and `THIRD_PARTY_NOTICES` from a
newer NAPA commit, update the commit and checksums above, and copy the two notices to
`ExternalLicenses/NAPA-NOTICE` and `ExternalLicenses/NAPA-THIRD_PARTY_NOTICES` (whose references to each other
carry the `NAPA-` prefix). WPILib's toolchains compile C with `-pedantic -Werror` at the compiler's default
standard, so check a new core with `gcc -std=gnu17 -pedantic -Werror -Wall -Wextra -fsyntax-only napa_core.c`, and
again with `-DNAPA_NO_SIMD` added (NAPA's CI runs the same checks).
