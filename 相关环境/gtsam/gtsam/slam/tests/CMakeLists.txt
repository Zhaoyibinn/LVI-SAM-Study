# if GTSAM_ENABLE_BOOST_SERIALIZATION is OFF then exclude some tests
if (NOT GTSAM_ENABLE_BOOST_SERIALIZATION)
  # create a semicolon seperated list of files to exclude
  set(EXCLUDE_TESTS "testSerializationDataset.cpp" "testSerializationInSlam.cpp")
  message(STATUS "Excluding tests: ${EXCLUDE_TESTS}")
else()
  set(EXCLUDE_TESTS "")
endif()

gtsamAddTestsGlob(slam "test*.cpp" "${EXCLUDE_TESTS}" "gtsam")
