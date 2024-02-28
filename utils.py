import zipfile
from enum import Enum
from pathlib import Path

import requests
from google.api_core.page_iterator import Iterator

from google.cloud import storage
from google.cloud.storage import Client, Bucket, Blob


def download_file_from_url(url: str, save_path: Path, chunk_size=128) -> None:
    r = requests.get(url, stream=True)
    with save_path.open('wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


def unzip_file(archive_path: Path, save_path: Path = None, delete_archive: bool = False) -> None:
    if save_path is None:
        save_path = archive_path
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(save_path)

    if delete_archive:
        archive_path.unlink()


class DataSets(Enum):
    modis_reflectance_500m = "MODIS/006/MCD43A4"
    landsat8_reflectance_30m = "LANDSAT/LC08/C01/T1_SR"


def download_file_from_gcs(bucket: str, filename: Path, save_path: Path):
    client: Client = storage.Client.from_service_account_json("secrets/export-access.json")

    bucket: Bucket = client.get_bucket(bucket)

    blob: Blob = bucket.get_blob(str(filename), timeout=30)
    if blob is not None and blob.exists(timeout=30):
        with save_path.open("wb") as save_fi:
            blob.download_to_file(save_fi)
    else:
        print(f"could not find blob {filename} in {bucket}")


def download_blob_gcs(blob: Blob, save_path: Path, timeout: int = 30):
    if blob is not None and blob.exists(timeout=timeout):
        with save_path.open("wb") as save_fi:
            blob.download_to_file(save_fi)


def download_fileset_from_gcs(bucket: str, filename_prefix: Path, save_path_prefix: Path, timeout: int = 30):
    client: Client = storage.Client.from_service_account_json("secrets/export-access.json")

    bucket: Bucket = client.get_bucket(bucket)
    blobs: Iterator[Blob] = bucket.list_blobs(prefix=str(filename_prefix), timeout=timeout)

    blob: Blob
    for blob in blobs:
        save_path: Path = save_path_prefix / Path(blob.name)
        download_blob_gcs(blob, save_path)


if __name__ == "__main__":
    prefix = Path("high_resolution_landsat8_30m_20200906T211544z")
    download_fileset_from_gcs("xcb-gee-exports", prefix, Path())
