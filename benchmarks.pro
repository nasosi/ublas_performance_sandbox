#-------------------------------------------------
#
# Project created by QtCreator 2013-03-26T11:28:25
#
#-------------------------------------------------

QT       -= core gui

TARGET = benchmarks
CONFIG   += console
CONFIG   -= app_bundle
QMAKE_CXXFLAGS_RELEASE -= -O2
QMAKE_CXXFLAGS_RELEASE += -DNDEBUG
QMAKE_LFLAGS_RELEASE -= -O1
QMAKE_CXXFLAGS += -std=c++11 -fopenmp
#QMAKE_CXXFLAGS_RELEASE += -DNDEBUG -O2 -ftree-vectorize -mavx -ftree-vectorizer-verbose=5
#QMAKE_CXXFLAGS_RELEASE += -DNDEBUG -O3
QMAKE_CXXFLAGS_RELEASE += -DNDEBUG -O3 -funroll-loops -mavx -ftree-vectorize
LIBS += -fopenmp
TEMPLATE = app

SOURCES += main.cpp

