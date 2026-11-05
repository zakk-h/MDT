import os
from pathlib import Path
from importlib.metadata import distribution
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11


class BuildExt(build_ext):
    c_opts = {
        "msvc": ["/O2", "/std:c++17"],
        "unix": ["-O3", "-DNDEBUG", "-funroll-loops"],
    }


    l_opts = {
        "msvc": [],
        "unix": [],
    }

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, []).copy()
        link_opts = self.l_opts.get(ct, []).copy()

        if ct == "unix":
            opts.append("-std=c++17")
            opts.append("-fPIC")

        aggressive = os.environ.get("AGGRESSIVE", "").lower() in ("1", "true", "yes")

        if aggressive and ct == "unix":
            opts += [
                "-march=core-avx-i",
                "-mtune=generic",
                "-flto",
            ]
            link_opts += [
                "-flto",
                "-lm",
            ]
            print("** Building separable with additional flags")
        elif aggressive and ct != "unix":
            print("Non-unix compiler detected; using safe flags.")

        for ext in self.extensions:
            ext_opts = opts.copy()
            if ext.name == "tree_again.clari_tree._core":
                if ct == "unix":
                    ext_opts = ["-O2" if flag == "-O3" else flag for flag in ext_opts]
                elif ct == "msvc":
                    ext_opts.append("/bigobj")
            ext.extra_compile_args = ext_opts
            ext.extra_link_args = link_opts

        build_ext.build_extensions(self)


eigen_include = os.environ.get("EIGEN3_INCLUDE_DIR")
if eigen_include is None:
    eigen_include = str(distribution("cmeel-eigen").locate_file("cmeel.prefix/include/eigen3"))
if not (Path(eigen_include) / "Eigen" / "Core").is_file():
    raise RuntimeError("Eigen headers not found; set EIGEN3_INCLUDE_DIR to the directory containing Eigen/Core.")

ext_modules = [
    Extension(
        "tree_again.weightedseparable._core",
        sources=["src/tree_again/weightedseparable/_core.cpp"],
        include_dirs=[
            pybind11.get_include(),
            "src/tree_again/weightedseparable/cpp",
        ],
        language="c++",
    ),
    Extension(
        "tree_again.clari_tree._core",
        sources=[
            "src/tree_again/clari_tree/cpp/bindings.cpp",
            "src/tree_again/clari_tree/cpp/clari_tree.cpp",
            "src/tree_again/clari_tree/cpp/clari_tree_const.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            eigen_include,
            "src/tree_again/clari_tree/include",
        ],
        language="c++",
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
)
