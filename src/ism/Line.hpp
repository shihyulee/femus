/*=========================================================================

 Program: FEMuS
 Module: Line
 Authors: Eugenio Aulisa and Giacomo Capodaglio

 Copyright (c) FEMuS
 All rights reserved.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notice for more information.

=========================================================================*/


#ifndef __femus_ism_Line_hpp__
#define __femus_ism_Line_hpp__

//----------------------------------------------------------------------------
// includes :
//----------------------------------------------------------------------------
#include "MarkerTypeEnum.hpp"
#include "ParallelObject.hpp"
#include "Mesh.hpp"
#include "Marker.hpp"

#include "vector"
#include "map"
#include "MyVector.hpp"

namespace femus {

  class Line : public ParallelObject {
    public:
      
    Line( const std::vector < std::vector < double > > x, 
	  const std::vector <MarkerType> &markerType,
          Mesh *mesh, const unsigned & solType);  
    ~Line();

      std::vector < std::vector < std::vector < double > > > GetLine() {
        return _line;
      }

      std::vector < Marker*> GetParticles() {
        return _particles;
      }


      void AdvectionParallel(Solution* sol, const unsigned &n, const double& T, const unsigned &order);

      void UpdateLine();

      

    private:
      std::vector < std::vector < std::vector < double > > > _line;

      std::vector < Marker*> _particles;
      std::vector < unsigned > _markerOffset;
      std::vector < unsigned > _printList;
      unsigned _size;
      unsigned _dim;

      static const double _a[4][4][4];
      static const double _b[4][4];
      static const double _c[4][4];

  };
} //end namespace femus



#endif



