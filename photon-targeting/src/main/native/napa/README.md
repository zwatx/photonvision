# NAPA

NAPA's C core, copied unchanged from [zwatx/NAPA](https://github.com/zwatx/NAPA) (`src/napa_vision/_native`):
branch `feat/consolidated-detector-review` at commit `8d5a569` (released as NAPA 0.6.0). It is compiled into `photontargetingJNI` and called
through `NapaDetectorJNI.cpp`. [SHA256SUMS](SHA256SUMS) lists every copied file; check them with
`sha256sum -c SHA256SUMS` in this folder.

NAPA is Copyright (c) 2026 Zachary Wilson and is used here under the GPL-3.0-only; see [NOTICE](NOTICE). The tag
family codewords come from AprilTag under its BSD license; see [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).

To update from a newer NAPA commit:

1. Copy `napa_core.c`, `napa_core.h`, `families.h`, `families/`, `NOTICE` and `THIRD_PARTY_NOTICES` here, and
   update the commit above.
2. Regenerate the checksums:
   `sha256sum napa_core.c napa_core.h families.h families/*.h NOTICE THIRD_PARTY_NOTICES > SHA256SUMS`.
3. Copy the two notices to `ExternalLicenses/NAPA-NOTICE` and `ExternalLicenses/NAPA-THIRD_PARTY_NOTICES`, adding
   the `NAPA-` prefix to their references to each other.
4. WPILib's toolchains compile C with `-pedantic -Werror` at the compiler's default standard, so check the new core
   with `gcc -std=gnu17 -pedantic -Werror -Wall -Wextra -fsyntax-only napa_core.c`, and again with `-DNAPA_NO_SIMD`
   added (NAPA's CI runs the same checks).
