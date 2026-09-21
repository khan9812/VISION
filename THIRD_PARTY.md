# Third-party software, models and data

The MIT license covers VISION-authored code and does not replace third-party terms.

- **SAM 2 / SAM 2.1:** [Meta repository](https://github.com/facebookresearch/sam2), pinned source commit `393ae336a752d26e68fb9a586e3d4ac14ff1e3c5`. Weights are not bundled.
- **CLIP:** [OpenAI CLIP](https://github.com/openai/CLIP), installed as `openai-clip==1.0.1`. Weights are not bundled.
- **BM3D:** [BM3D package](https://pypi.org/project/bm3d/4.0.3/), installed as a dependency; its package and binary-library terms apply independently.
- **Noise2SR:** adapted code in `modules/noise2sr/`, from [ZS-Denoiser-HREM](https://github.com/MeijiTian/ZS-Denoiser-HREM). Its [MIT LICENSE](https://github.com/MeijiTian/ZS-Denoiser-HREM/blob/87b457a4f31aa34b0d5255092deba3ad5cbaea26/LICENSE), including Xuanyu Tian's copyright notice, is preserved in [licenses/Noise2SR-LICENSE.txt](licenses/Noise2SR-LICENSE.txt).

Noise2SR's upstream README also contains a research/education and noncommercial usage notice ([Usage, section 4](https://github.com/MeijiTian/ZS-Denoiser-HREM#4-usage)). Both are disclosed; VISION does not resolve their different wording or expand upstream permissions. Commercial users should obtain clarification from the upstream rights holder.

Noise2SR adaptations include a callable wrapper, package-qualified imports, explicit settings/seeds/data-loader options, device selection, normalization/padding, and skip-connection size alignment. This is an adapted distribution.

Source microscopy images, annotations and literature figure excerpts retain their original rights. The included composite manuscript figures and numerical tables do not grant MIT redistribution rights to their underlying images. Full-resolution datasets and weights are excluded; see [DATA.md](DATA.md).

Cite the original SAM 2, CLIP, BM3D and Noise2SR publications alongside VISION. Scientific references are provided in the manuscript bibliography.
