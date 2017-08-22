package(
    default_visibility = [
        ":nli",
        "//word_context:hypertext",
    ]
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package_group(
    name = "nli",
    packages = [
        "//nli/...",
    ],
)

py_library(
    name = "eval_nli",
    srcs = ["eval_nli.py"],
    deps = [
        ":mlp_classifier",
        ":tfrecord_creator",
    ],
)

py_test(
    name = "eval_nli_test",
    srcs = ["eval_nli_test.py"],
    data = glob(["testdata/*"]),
    deps = [
        ":baseline_encoders",
        ":eval_nli",
    ],
)

py_library(
    name = "baseline_encoders",
    srcs = ["baseline_encoders.py"],
)

py_test(
    name = "baseline_encoders_test",
    srcs = ["baseline_encoders_test.py"],
    deps = [
        ":baseline_encoders",
    ],
)

py_library(
    name = "mlp_classifier",
    srcs = ["mlp_classifier.py"],
)

py_test(
    name = "mlp_classifier_test",
    srcs = ["mlp_classifier_test.py"],
    data = glob(["testdata/*"]),
    deps = [
        ":mlp_classifier",
        ":tfrecord_creator",
    ],
)

py_library(
    name = "tfrecord_creator",
    srcs = ["tfrecord_creator.py"],
)

py_test(
    name = "tfrecord_creator_test",
    srcs = ["tfrecord_creator_test.py"],
    deps = [
        ":tfrecord_creator",
    ],
)
