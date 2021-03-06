INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_HYPERSPACE_COMM hyperspace_comm)

FIND_PATH(
    HYPERSPACE_COMM_INCLUDE_DIRS
    NAMES hyperspace_comm/api.h
    HINTS $ENV{HYPERSPACE_COMM_DIR}/include
        ${PC_HYPERSPACE_COMM_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    HYPERSPACE_COMM_LIBRARIES
    NAMES gnuradio-hyperspace_comm
    HINTS $ENV{HYPERSPACE_COMM_DIR}/lib
        ${PC_HYPERSPACE_COMM_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(HYPERSPACE_COMM DEFAULT_MSG HYPERSPACE_COMM_LIBRARIES HYPERSPACE_COMM_INCLUDE_DIRS)
MARK_AS_ADVANCED(HYPERSPACE_COMM_LIBRARIES HYPERSPACE_COMM_INCLUDE_DIRS)

