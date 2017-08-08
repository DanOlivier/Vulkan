file(GLOB IMGUI_SOURCE ../external/imgui/*.cpp)

buildExample(imgui main.cpp ${IMGUI_SOURCE})
