#pragma once
#include <vector>
#include "glm.hpp"
#define  CHUNKSIZE_X 1
#define  CHUNKSIZE_Y 1
#define  CHUNKSIZE_Z 1

enum class Faces
{
	TOP, BOT, LEFT, RIGHT, FRONT, BACK
};

class Chunk
{
public:
	Chunk();
	void GenerateFace(Faces dir);
	std::vector<int>& GetIndices();
	std::vector<glm::vec3>& GetVertices();

protected:

private:
	std::vector<int>m_Indices;
	std::vector<glm::vec3> m_Vertices;
	int m_IndicesIndex{0};
};