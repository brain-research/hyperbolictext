package(default_visibility = [":hypertext"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package_group(
    name = "hypertext",
    packages = [
        "//word_context/...",
    ],
    includes = [
        "//skip_thoughts:internal",
        "//skip_thoughts_dist:distributed",
        "//nli:nli",
    ],
)

py_library(
    name = "word_context_encoder",
    srcs = ["word_context_encoder.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":word_context_model",
        "//skip_thoughts:skip_thoughts_encoder",
    ],
)

py_library(
    name = "word_context_model",
    srcs = ["word_context_model.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":metrics",
        "//skip_thoughts:skip_thoughts_model",
        "//skip_thoughts/ops:input_ops",
    ],
)

py_library(
    name = "configuration",
    srcs = ["configuration.py"],
    srcs_version = "PY2AND3",
)

py_binary(
    name = "track_perplexity",
    srcs = ["track_perplexity.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":configuration",
        ":word_context_model",
        "//skip_thoughts_dist/ops:evaluation",
    ],
)

py_binary(
    name = "evaluate_trec",
    srcs = ["evaluate_trec.py"],
    deps = [
        ":configuration",
        ":tools",
        ":word_context_encoder",
        "//nli:eval_nli",
    ],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":configuration",
        ":word_context_model",
        "//skip_thoughts_dist/ops:training",
    ],
)

py_library(
    name = "metrics",
    srcs = ["metrics.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "metrics_test",
    size = "small",
    srcs = ["metrics_test.py"],
    deps = [
        ":metrics",
    ],
)

py_binary(
    name = "post_analysis",
    srcs = ["post_analysis.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":configuration",
        ":word_context_model",
        ":tools",
    ],
)

py_binary(
    name = "test_run",
    srcs = ["test_run.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":configuration",
        ":word_context_model",
        "//skip_thoughts_dist/ops:training",
    ],
)

py_library(
    name = "tools",
    srcs = ["tools.py"],
    srcs_version = "PY2AND3",
)
