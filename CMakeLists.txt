cmake_minimum_required(VERSION 3.9)
project(Threads_IVQNet2)

set(CMAKE_CXX_STANDARD 11)

add_executable(Threads_IVQNet2
        Threads_IVQNet2/Agent.cpp
        Threads_IVQNet2/Agent.hpp
        Threads_IVQNet2/Contingence.cpp
        Threads_IVQNet2/Contingence.hpp
        Threads_IVQNet2/Data.cpp
        Threads_IVQNet2/Data.hpp
        Threads_IVQNet2/ErrorManager.cpp
        Threads_IVQNet2/ErrorManager.hpp
        Threads_IVQNet2/FileManager.cpp
        Threads_IVQNet2/FileManager.hpp
        Threads_IVQNet2/Helper.cpp
        Threads_IVQNet2/Helper.hpp
        Threads_IVQNet2/main.cpp
        Threads_IVQNet2/NeuralNetwork.cpp
        Threads_IVQNet2/NeuralNetwork.hpp
        Threads_IVQNet2/Param.cpp
        Threads_IVQNet2/Param.hpp
        Threads_IVQNet2/State.cpp
        Threads_IVQNet2/State.hpp)
