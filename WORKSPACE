workspace(name = "any2json")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "bazel_features",
    strip_prefix = "bazel_features-1.20.0",
    url = "https://github.com/bazel-contrib/bazel_features/releases/download/v1.20.0/bazel_features-v1.20.0.tar.gz",
)

load("@bazel_features//:deps.bzl", "bazel_features_deps")
bazel_features_deps()

http_archive(
    name = "rules_oci",
    strip_prefix = "rules_oci-2.0.0",
    url = "https://github.com/bazel-contrib/rules_oci/releases/download/v2.0.0/rules_oci-v2.0.0.tar.gz",
)

load("@rules_oci//oci:dependencies.bzl", "rules_oci_dependencies")

rules_oci_dependencies()

load("@rules_oci//oci:repositories.bzl", "oci_register_toolchains")

oci_register_toolchains(name = "oci")

load("@rules_oci//oci:pull.bzl", "oci_pull")

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
