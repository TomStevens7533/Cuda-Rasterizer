#pragma once
#include "glm.hpp"
#include <memory>
#include <vector>
#include "BlockType.h"	
#include "Content/PerlinNoise.hpp"

#define  CHUNKSIZE_X 128
#define  CHUNKSIZE_Z 128
#define  CHUNKSIZE_Y 128
#define  CHUNKSIZE_Y_MAX_TERRAIN 120
#define  CHUNKSIZE_Y_MIN_TERRAIN 10


#define  MAX_DEPTH 8
enum class Faces
{
	//Positive
	//0	  1	   2	
	TOP, RIGHT, FRONT,
	//Negative
	//3      4     5 
	BOT, LEFT, BACK

};
struct SVOBaseNode {
	virtual ~SVOBaseNode() = default;
};
struct SVOInnerNode final : public SVOBaseNode{
	SVOInnerNode() = default;
	~SVOInnerNode() {
		for (size_t i = 0; i < 8; i++)
			delete children[i];
	}
	// Pointers to this node's children (if it is an internal node)
	SVOBaseNode* children[2][2][2]; //X Y Z
	SVOInnerNode* pParentNode = nullptr;
};
struct SVOLeafNode final : public SVOBaseNode{
	// Data for the voxel represented by this node
	glm::vec3 data;
	BlockTypes blockID;
	SVOInnerNode* pParentNode = nullptr;
};
class ChunkMesh final
{
public:
	ChunkMesh();
	~ChunkMesh();
	void TraverseSVO();
	std::vector<glm::vec3>& GetVertices();
	std::vector<int>& GetIndices();


private:
	void FillSVO();
	void FillSVONode(SVOBaseNode* childToFill, int depth, int resolution, int xPos, int yPos, int zPos);
	BlockTypes GetTerrainData(glm::vec3 position);
	void GenerateFace(Faces dir, glm::vec3 position);
	void TraverseSVONode(SVOBaseNode* pNode, int depth);
	void CheckGenerationOfFace(Faces dir, SVOLeafNode* currLeafnode);

private:
	SVOInnerNode* m_StarterNode = nullptr;
	std::vector<int>m_Indices;
	std::vector<glm::vec3> m_Vertices;
	int m_IndicesIndex{ 0 };
	int m_blockdetected{ 0 };
	int m_Facedetected{ 0 };

	const siv::PerlinNoise perlin{ time(NULL)};

};
