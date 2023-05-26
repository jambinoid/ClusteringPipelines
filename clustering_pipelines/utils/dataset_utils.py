from typing import Generator, Iterable, Literal, Union
from pathlib import Path
import urllib.request

import tarfile
import zipfile


def _get_members_stripped(arch, n_folders_stripped = 1):
    members = []
    for member in arch.getmembers():
        p = Path(member.path)
        member.path = p.relative_to(*p.parts[:n_folders_stripped])
        members.append(member)
    return members


def load_and_extract(
    url: str,
    dest: Union[str, Path],
    archive_type: Literal["tar", "tar.gz", "zip"],
    strip: int | None = None
) -> None:
    """
    Load archive from given url and extract it to given directory

    Params:
        url (str): string with url to load archive
        dest (str or pathlib.Path): path ot directory where to extract
            loaded archive
    
    Raises:
        NotImplementedError: if file is not an archive or archive type is not supported

    """
    # TODO: add progress bar to 'urlretrieve' as 'reporthook' argument
    filehandle, _ = urllib.request.urlretrieve(url)
    if archive_type == "tar":
        arch = tarfile.open(filehandle, "r:")
    elif archive_type == "tar.gz":
        arch = tarfile.open(filehandle, "r:gz")
    elif archive_type == "zip":
        arch = zipfile.ZipFile(filehandle, 'r')
    else:
        raise NotImplementedError(
            f"file {filehandle} is not an archive or archive type is not supported")
    
    members = None if strip is None else _get_members_stripped(arch, strip)
    arch.extractall(path=dest, members=members)
    
    arch.close()


def chunkify(data: Iterable, chunk_size: int) -> Generator:
    """
    Split iterable into chunks.

    Args:
        data (iterable): iterable to split into chunks.
        chunk_size (int): size of a chunk.

    Return:
        generator
    
    """
    for id in range(0, len(data), chunk_size):
        yield data[id:id+chunk_size]
