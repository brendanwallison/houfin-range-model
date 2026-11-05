# E081 -- the age model on the new DESK (desk-temporal E086w, cbc1902) (pre-registered 2026-10-23)

User: start using the new DESK versions now (likely final: an ensemble trained on CBC + BBS). The first completed build is
E086w's CBC-supervised DESK trained from 1902 (encoder/cube_desk_cbc1902; 64-D, Ruzicka contract, 1902-2025). NOTE: its
handoff note was marked HOLD pending E086x/E086v local checks; these runs test the PIPELINE and give a first read, not a
verdict on that DESK version. Its matched control (ctl1902) is still training; an E081 control arm follows when it lands.
Build (task "build_cbc1902"): path features -> model ingest at width 64 -> numpyro_input_cbc1902_r64 (isolated paths;
same dispersal spec as vl_r64; route ids / subcell strata copied from vl_r64 -- they do not depend on DESK).
Arms (holdout mask, smooth Allee k = 3; each mirrors a production-DESK twin):
- 81a = 79b (split r32, free temporal change)   - 81b = 80g (tied space-for-time .14)   - 81c = 80d (static)
Read: twin-by-twin -- held-out density, demography, plains corridor at MAP vs refined and by era, temporal lambda swing,
deviation/level ratios. The temporal swing and the corridor are the main questions: does a DESK supervised over 1902-2025
change less implausibly over time than production?
- Added (user, 2026-10-23): the matched-control DESK (ctl1902; no CBC terms) gets the same three arms -- 81d/81e/81f =
  twins of 81a/81b/81c -- isolating what CBC supervision changes. Its build is enqueued when its cube exists
  (cube_desk_ctl1902/cube_meta.json); the ctl1902 DESK training was orphaned once by the dev 2 h limit.
- NOT run on the age model: the E086z seed fits (tempho T0 1996 + winter sum kernel). They hold out 1966-1995 from DESK
  training and add 24 private winter dims -- DESK validation fits, not production builds. The production ensemble
  members get builds when trained on the full window (ensemble = mean kernel: stack members' features / sqrt(n), top-64
  eigenfeatures).
