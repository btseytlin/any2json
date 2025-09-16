# workspace
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_oci",
    sha256 = "e96d70faa4bace3e09fdb1d7d1441b838920f491588889ff9a7e2615afca5799",
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
    digest = "sha256:bf2d42c1240bb8d3e87cce9b2a16c0e18c691e2e8b6b55f0063b55696292d6d0",
    platforms = [
        "linux/amd64",
        "linux/arm64",
    ]
)