cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(VulkanRewriter)

set(CMAKE_CXX_FLAGS "-std=c++11 -fno-rtti -O0 -g")

# LLVM_SRC_PATH is the path to the root of the checked out source code. This
# directory should contain the configure script, the include/ and lib/
# directories of LLVM, Clang in tools/clang/, etc.

set(LLVM_SRC_PATH $ENV{HOME}/tools/llvm)

# LLVM_BUILD_PATH is the directory in which you built LLVM - where you ran
# configure or cmake.
# For linking vs. a binary build of LLVM, point to the main untarred directory.
# LLVM_BIN_PATH is the directory where binaries are placed by the LLVM build
# process. It should contain the tools like opt, llc and clang. The default
# reflects a release build with CMake and Ninja. binary build of LLVM, point it
# to the bin/ directory.
set(LLVM_BUILD_PATH $ENV{HOME}/tools/llvm/build)
set(LLVM_BIN_PATH ${LLVM_BUILD_PATH}/bin)

message(STATUS -----------------------------------------------)
message(STATUS "Using LLVM_SRC_PATH = ${LLVM_SRC_PATH}")
message(STATUS "Using LLVM_BUILD_PATH = ${LLVM_BUILD_PATH}")
message(STATUS "Using LLVM_BIN_PATH = ${LLVM_BIN_PATH}")
message(STATUS -----------------------------------------------)

execute_process(OUTPUT_VARIABLE LLVM_CXXFLAGS 
    COMMAND ${LLVM_BIN_PATH}/llvm-config --cxxflags
)
string(REGEX REPLACE "\n$" "" LLVM_CXXFLAGS ${LLVM_CXXFLAGS})
string(REGEX REPLACE "\n" " " LLVM_CXXFLAGS ${LLVM_CXXFLAGS})

execute_process(OUTPUT_VARIABLE LLVM_LDFLAGS
    COMMAND ${LLVM_BIN_PATH}/llvm-config --ldflags --libs --system-libs
)
string(REGEX REPLACE "\n$" "" LLVM_LDFLAGS ${LLVM_LDFLAGS})
string(REGEX REPLACE "\n" " " LLVM_LDFLAGS ${LLVM_LDFLAGS})

# Plugins shouldn't link LLVM and Clang libs statically, because they are
# already linked into the main executable (opt or clang). LLVM doesn't like its
# libs to be linked more than once because it uses globals for configuration
# and plugin registration, and these trample over each other.
execute_process(OUTPUT_VARIABLE LLVM_LDFLAGS_NOLIBS
    COMMAND ${LLVM_BIN_PATH}/llvm-config --ldflags
)
string(REGEX REPLACE "\n$" "" LLVM_LDFLAGS_NOLIBS ${LLVM_LDFLAGS_NOLIBS})
string(REGEX REPLACE "\n" " " LLVM_LDFLAGS_NOLIBS ${LLVM_LDFLAGS_NOLIBS})

set(PLUGIN_CXXFLAGS -fpic)
set(PLUGIN_LDFLAGS -shared)

#message(STATUS "Using LLVM_CXXFLAGS = ${LLVM_CXXFLAGS}")
#message(STATUS "Using LLVM_LDFLAGS = ${LLVM_LDFLAGS}")
#message(STATUS "Using LLVM_LDFLAGS_NOLIBS = ${LLVM_LDFLAGS_NOLIBS}")

# These are required when compiling vs. a source distribution of Clang. For
# binary distributions llvm-config --cxxflags gives the right path.
set(CLANG_INCLUDES
	-I${LLVM_SRC_PATH}/tools/clang/include
    -I${LLVM_BUILD_PATH}/tools/clang/include
)

# List of Clang libraries to link. The proper -L will be provided by the
# call to llvm-config
# Note that I'm using -Wl,--{start|end}-group around the Clang libs; this is
# because there are circular dependencies that make the correct order difficult
# to specify and maintain. The linker group options make the linking somewhat
# slower, but IMHO they're still perfectly fine for tools that link with Clang.
set(CLANG_LIBS 
	-Wl,--start-group
	-lclangAST
	-lclangASTMatchers
	-lclangAnalysis
	-lclangBasic
	-lclangDriver
	-lclangEdit
	-lclangFrontend
	-lclangFrontendTool
	-lclangLex
	-lclangParse
	-lclangSema
	-lclangEdit
	-lclangRewrite
	-lclangRewriteFrontend
	-lclangStaticAnalyzerFrontend
	-lclangStaticAnalyzerCheckers
	-lclangStaticAnalyzerCore
	-lclangSerialization
	-lclangToolingCore
	-lclangTooling
	-lclangFormat
	-Wl,--end-group
)

add_executable(Rewriter1 Rewriter1.cpp)

target_compile_options(Rewriter1 PUBLIC ${PLUGIN_CXXFLAGS})
target_link_libraries(Rewriter1 ${PLUGIN_LDFLAGS})

#target_compile_options(Rewriter1 PUBLIC ${CLANG_INCLUDES})
#target_compile_options(Rewriter1 PUBLIC ${LLVM_CXXFLAGS})
#target_link_libraries(Rewriter1 ${CLANG_LIBS})
#target_link_libraries(Rewriter1 ${LLVM_LDFLAGS})

