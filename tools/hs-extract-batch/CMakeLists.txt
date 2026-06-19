set(TARGET llama-hs-extract-batch)
add_executable(${TARGET} hs-extract-batch.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama)
