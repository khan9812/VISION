# Initial manuscript release preparation

VISION provides projected particle area, projected morphology and PF-SUI analysis with SAM 2.1 and CLIP. This package includes the application, executable settings, archived numerical observations and statistical reproduction scripts.

The preparation corrects Noise2SR patch/batch settings and their cache/export records, passes the final preprocessed image to GUI CLIP classification, shares crop/centroid filtering behavior, reports unavailable PF-SUI when its sample is insufficient, and rebuilds tables and plots after manual particle deletion. It includes the restored Fig. 5, final Fig. 7c and Fig. 12, an accurate screening schematic, and an unclipped Fig. 7b CI.

Installation requirements resolve the known pandas/Streamlit constraint conflict and use a pinned official SAM source archive. New-environment installation still needs confirmation; see [verification scope](RELEASE_VERIFICATION.md). Archived scientific results were preserved, and the complete neural benchmark was not rerun.

Raw images, prepared crops, annotation masks and model weights are distributed separately. See [DATA.md](DATA.md) and [THIRD_PARTY.md](THIRD_PARTY.md) for availability and terms. Assign the actual release tag/date and repository URL when publishing.
