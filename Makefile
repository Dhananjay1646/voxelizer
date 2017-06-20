CC = g++
CFLAGS = -std=c++11 -Wall -O3
EIGEN_INCLUDE = -I eigen/
LIBS = -lblas 

all: src/main.cpp Voxelizer.o triangleCube.o
	$(CC) $(CFLAGS) $(EIGEN_INCLUDE) $(LIBS) Voxelizer.o triangleCube.o src/main.cpp -I src/ -o voxelizer
	mv voxelizer bin

Voxelizer.o:
	$(CC) $(CFLAGS) $(EIGEN_INCLUDE) $(LIBS) -I src/ -g -c src/Voxelizer.cpp

triangleCube.o:
	$(CC) $(CFLAGS) $(EIGEN_INCLUDE) $(LIBS) -I src/ -g -c src/triangleCube.c
#
#PointCloudSorter.o:
#	$(CC) $(CFLAGS) $(EIGEN_INCLUDE) $(LIBS) -I src/ -g -c src/PointCloudSorter.cpp
#
#PCA.o:
#	$(CC) $(CFLAGS) $(EIGEN_INCLUDE) $(LIBS) -I src/ -g -c src/PCA.cpp

clean:
	rm -f *.o
