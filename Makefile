CXX  = g++
CPPS = main.cpp cbir/*.cpp 
CONF = `pkg-config --cflags --libs opencv4`
matcher: main.cpp
	$(CXX) -std=c++14 -o matcher $(CPPS) $(CONF)
