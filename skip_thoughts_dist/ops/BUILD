package(
    default_visibility = [
        "//skip_thoughts_dist:distributed",
        "//word_context:hypertext",
    ]
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "training",
    srcs = ["training.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "evaluation",
    srcs = ["evaluation.py"],
    srcs_version = "PY2AND3",
)
