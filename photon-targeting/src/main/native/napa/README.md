# NAPA

NAPA's C core, copied unchanged from [zwatx/NAPA](https://github.com/zwatx/NAPA) at commit `3424f56` (v0.5.1)
(`src/napa_vision/_native`). It is compiled into `photontargeting` and called through
`NapaDetectorJNI.cpp`.

NAPA is Copyright (c) 2026 Zachary Wilson and is used here under the GPL-3.0-only; see [NOTICE](NOTICE). The tag
family codewords come from AprilTag under its BSD license; see [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).

To update, copy `napa_core.c`, `napa_core.h`, `families.h` and `families/` from a newer NAPA commit.
