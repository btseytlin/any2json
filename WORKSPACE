# workspace
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_oci",
    strip_prefix = "rules_oci-2.0.0-alpha2",
    url = "https://github.com/bazel-contrib/rules_oci/releases/download/v2.0.0-alpha2/rules_oci-v2.0.0-alpha2.tar.gz",
)

load("@rules_oci//oci:dependencies.bzl", "rules_oci_dependencies")

rules_oci_dependencies()

load("@rules_oci//oci:repositories.bzl", "oci_register_toolchains")

oci_register_toolchains(name = "oci")

# You can pull your base images using oci_pull like this:
load("@rules_oci//oci:pull.bzl", "oci_pull")

# Pull the base OCI image
oci_pull(
    name = "base_image",
    registry = "index.docker.io",
    repository = "runpod/pytorch",
    tag = "2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
    platforms = [
        "linux/amd64",
        "linux/arm64",
    ]
)