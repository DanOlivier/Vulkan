file(GLOB SOURCE *.cpp *.cpp)
set(SOURCE
    #VulkanAndroid.cpp
    VulkanDebug.cpp
    VulkanExampleBase.cpp
    VulkanTools.cpp
)

add_library(VulkanBase ${SOURCE})

compile_shaders(VulkanBase
    shaders/textoverlay.vert
    shaders/textoverlay.frag
)
