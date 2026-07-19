#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
dataset_dir="${project_root}/Backend/data/sample_speech"
asset_base="https://download.pytorch.org/torchaudio/tutorial-assets"

mkdir -p "${dataset_dir}"

curl --fail --location --retry 3 \
  --output "${dataset_dir}/voices_speech_16khz.wav" \
  "${asset_base}/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
curl --fail --location --retry 3 \
  --output "${dataset_dir}/voices_speech_8khz.wav" \
  "${asset_base}/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042-8000hz.wav"

expected_16khz="c65fcd726d6b08c82c1e5dc7558f863cd8d483e3ed2f4a7bcf271dc1865ada14"
expected_8khz="17c4b9935a0d336478d97a60c3f737e80cdb4cc334c9b411050a8e32e2c2ec88"
actual_16khz="$(shasum -a 256 "${dataset_dir}/voices_speech_16khz.wav" | awk '{print $1}')"
actual_8khz="$(shasum -a 256 "${dataset_dir}/voices_speech_8khz.wav" | awk '{print $1}')"

if [[ "${actual_16khz}" != "${expected_16khz}" || "${actual_8khz}" != "${expected_8khz}" ]]; then
  echo "Sample dataset checksum verification failed." >&2
  exit 1
fi

cp "${script_dir}/sample-dataset/sample_speech_metadata.csv" "${dataset_dir}/sample_speech_metadata.csv"
cp "${script_dir}/sample-dataset/ATTRIBUTION.md" "${dataset_dir}/ATTRIBUTION.md"

echo "Sample speech dataset installed in ${dataset_dir}"
