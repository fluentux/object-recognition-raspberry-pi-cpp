# Find all *.cpp and *.hpp files (store them in variables SOURCE_FILES and HEADER_FILES)
FILE(GLOB SOURCE_FILES [^\\.]*.cpp)
FILE(GLOB HEADER_FILES [^\\.]*.hpp)

# Define SOURCES variable so that it contains all the files
set(SOURCES ${SOURCE_FILES} ${HEADER_FILES})

# Set default compile flags for GCC
if(CMAKE_COMPILER_IS_GNUCXX)
	message(STATUS "GCC detected, enabling pedantic mode and warnings")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++98 -pedantic -Wall -Wextra")
endif()

# Generate config.h from config.h.in
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/config.h.in" "${CMAKE_CURRENT_BINARY_DIR}/config.h" @ONLY)
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

# Libraries
find_package(raspicam REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(LIBS ${LIBS} ${OpenCV_LIBRARIES} ${raspicam_CV_LIBS})

# Build an executable named the same as the project short name.
# If you need more, you'll need to alter this script by repeating the following
# three lines for each executable (with the necessary changes)
add_executable("${PROJECT_SHORT_NAME}" ${SOURCES})
target_link_libraries("${PROJECT_SHORT_NAME}" ${LIBS})
install(TARGETS "${PROJECT_SHORT_NAME}" DESTINATION bin)

