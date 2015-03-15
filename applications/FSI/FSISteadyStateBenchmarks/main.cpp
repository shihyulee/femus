#include "MultiLevelProblem.hpp"
#include "NumericVector.hpp"
#include "Fluid.hpp"
#include "Solid.hpp"
#include "Parameter.hpp"
#include "FemusInit.hpp"
#include "SparseMatrix.hpp"
#include "FElemTypeEnum.hpp"
#include "ParsedFunction.hpp"
#include "InputParser.hpp"
#include "Files.hpp"
#include "MonolithicFSINonLinearImplicitSystem.hpp"
#include "../include/IncompressibleFSIAssembly.hpp"

using namespace std;
using namespace femus;

void AssembleMatrixResFSI(MultiLevelProblem &ml_prob, unsigned level, const unsigned &gridn, const bool &assemble_matrix);

bool SetBoundaryConditionTurek(const double &x, const double &y, const double &z,const char name[], 
		double &value, const int FaceName, const double = 0.);
bool SetBoundaryConditionDrum(const double &x, const double &y, const double &z,const char name[], 
		double &value, const int FaceName, const double = 0.);
bool SetBoundaryConditionBatheCylinder(const double &x, const double &y, const double &z,const char name[], 
				       double &value, const int facename, const double time);

bool SetBoundaryConditionBatheShell(const double &x, const double &y, const double &z,const char name[], 
				       double &value, const int facename, const double time);
bool SetBoundaryConditionComsol(const double &x, const double &y, const double &z,const char name[], 
		double &value, const int FaceName, const double = 0.);

 
bool SetRefinementFlag(const double &x, const double &y, const double &z, const int &ElemGroupNumber,const int &level);

void show_usage()
{
  std::cout << "Use --inputfile variable to set the input file" << std::endl;
  std::cout << "e.g.: ./Poisson --inputfile $FEMUS_DIR/applications/Poisson/input/input.json" << std::endl;
}

//------------------------------------------------------------------------------------------------------------------

int main(int argc, char *argv[]) {
  
  std::string path;

  if(argc < 2)
  {
    std::cout << argv[0] << ": You must specify the input file" << std::endl;
    show_usage();
    return 1;
  }

  for (int count = 1; count < argc; ++count)
  {
    std::string arg = argv[count];

    if ((arg == "-h") || (arg == "--help")) {
      show_usage();
      return 0;
    }
    else if ((arg == "-i") || (arg == "--inputfile"))
    {
      if (count + 1 < argc) {
        path = argv[++count];
      }
      else
      {
        std::cerr << "--input file option requires one argument." << std::endl;
        return 1;
      }
    }
  }
  
  /// Init Petsc-MPI communicator
  FemusInit mpinit(argc,argv,MPI_COMM_WORLD);

  //Files files; 
  //files.CheckIODirectories();
  //files.RedirectCout();
  
  // input parser pointer
  std::auto_ptr<InputParser> inputparser = InputParser::build(path);
  
  unsigned simulation = 0;
  // mesh ----------------------------------------
  double Lref, Uref;
  Lref = 1.; Uref = 1.;
  Parameter par(Lref,Uref);
  
  unsigned short nm,nr;
  unsigned int nlevels = inputparser->getValue("multilevel_problem.multilevel_mesh.first.system.fsi.linear_solver.type.multigrid.nlevels",1);   
  nm=nlevels;

  nr=0;

  int tmp=nm;
  nm+=nr;
  nr=tmp;

  
  MultiLevelMesh ml_msh;
  
  if(inputparser->isTrue("multilevel_mesh.first.type","filename"))
  {
    std::string filename = inputparser->getValue("multilevel_mesh.first.type.filename", "./input/input.neu"); 
    ml_msh.ReadCoarseMesh(filename.c_str(),"fifth",Lref);
    
    // it should be used until boundary condition will be red from file
    if(filename.compare("input/turek.neu") == 0){
      simulation = 1;
    }
    else if(filename.compare("input/beam.neu") == 0){
      simulation = 2;
    }  
    else if(filename.compare("input/drum.neu") == 0){
      simulation = 3;
    }  
    else if(filename.compare("input/bathe_FSI.neu") == 0) {
      simulation = 4;
    }
    else if(filename.compare("input/bathe_shell.neu") == 0){
      simulation = 5;
    }
    else if(filename.compare("input/bathe_cylinder.neu") == 0){
      simulation = 6;
    }
    else if(filename.compare("input/comsolbenchmark.neu") == 0){
      simulation = 7;
    }
  }
  else if(inputparser->isTrue("multilevel_mesh.first.type","box"))
  {
    int numelemx = inputparser->getValue("multilevel_mesh.first.type.box.nx", 2);
    int numelemy = inputparser->getValue("multilevel_mesh.first.type.box.ny", 2);
    int numelemz = inputparser->getValue("multilevel_mesh.first.type.box.nz", 0);
    double xa = inputparser->getValue("multilevel_mesh.first.type.box.xa", 0.);
    double xb = inputparser->getValue("multilevel_mesh.first.type.box.xb", 1.);
    double ya = inputparser->getValue("multilevel_mesh.first.type.box.ya", 0.);
    double yb = inputparser->getValue("multilevel_mesh.first.type.box.yb", 1.);
    double za = inputparser->getValue("multilevel_mesh.first.type.box.za", 0.);
    double zb = inputparser->getValue("multilevel_mesh.first.type.box.zb", 0.);
    ElemType elemtype = inputparser->getValue("multilevel_mesh.first.type.box.elem_type", QUAD9);
    ml_msh.GenerateCoarseBoxMesh(numelemx,numelemy,numelemz,xa,xb,ya,yb,za,zb,elemtype,"fifth");
  }
  else
  {
    std::cerr << "Error: no input mesh specified. Please check to have added the keyword mesh in the input json file! " << std::endl;
    return 1;
  }
  ml_msh.RefineMesh(nm,nr, SetRefinementFlag);
    
  ml_msh.PrintInfo();
  
  // it should be taken from multilevel_mesh
  unsigned int dimension = ml_msh.GetLevel(0)->GetDimension();
  
  // end-mesh----------------------------------------------------------------
      
  MultiLevelSolution ml_sol(&ml_msh);
  
   //Start System Variables
   ml_sol.AddSolution("DX",LAGRANGE,SECOND,1);
   ml_sol.AddSolution("DY",LAGRANGE,SECOND,1);
   if (dimension == 3) 
     ml_sol.AddSolution("DZ",LAGRANGE,SECOND,1);
   ml_sol.AddSolution("U",LAGRANGE,SECOND,1);
   ml_sol.AddSolution("V",LAGRANGE,SECOND,1);
   if (dimension == 3) 
     ml_sol.AddSolution("W",LAGRANGE,SECOND,1);
   // Pair each velocity varible with the corresponding displacement variable
   ml_sol.PairSolution("U","DX"); // Add this line
   ml_sol.PairSolution("V","DY"); // Add this line
   if (dimension == 3) 
     ml_sol.PairSolution("W","DZ"); // Add this line
   // Since the Pressure is a Lagrange multiplier it is used as an implicit variable
   ml_sol.AddSolution("P",DISCONTINOUS_POLYNOMIAL,FIRST,1);
   //ml_sol.AddSolution("P",LAGRANGE,FIRST,1);
   ml_sol.AssociatePropertyToSolution("P","Pressure"); // Add this line
   
   //Initialize (update Init(...) function)
   ml_sol.Initialize("All");
  
//   //Start System Variables
//   ml_sol.AddSolution("DX",LAGRANGE,SECOND,1);
//   ml_sol.AddSolution("DY",LAGRANGE,SECOND,1);
//   if (dimension == 3) 
//     ml_sol.AddSolution("DZ",LAGRANGE,SECOND,1);
//   ml_sol.AssociatePropertyToSolution("DX","Displacement"); // Add this line
//   ml_sol.AssociatePropertyToSolution("DY","Displacement"); // Add this line 
//   if (dimension == 3) 
//     ml_sol.AssociatePropertyToSolution("DZ","Displacement"); // Add this line 
//   ml_sol.AddSolution("U",LAGRANGE,SECOND,1);
//   ml_sol.AddSolution("V",LAGRANGE,SECOND,1);
//   if (dimension == 3) 
//     ml_sol.AddSolution("W",LAGRANGE,SECOND,1);
//   // Since the Pressure is a Lagrange multiplier it is used as an implicit variable
//   ml_sol.AddSolution("P",DISCONTINOUS_POLYNOMIAL,FIRST,1);
//   ml_sol.AssociatePropertyToSolution("P","Pressure"); // Add this line
// 
//   //Initialize (update Init(...) function)
//   ml_sol.Initialize("All");

   //Set Boundary (update Dirichlet(...) function)
   // New method
   //Set Boundary (update Dirichlet(...) function)
   ml_sol.InitializeBdc();
   
   // These information must be retrieved from MultiLevelSolution
   int numvar = 5;
   std::string varname[7] = {"DX","DY","U","V","P","NULL","NULL"};
   
   if (dimension == 3) {
     numvar = 7;
     varname[0] = "DX";  varname[0] = "DY";  varname[0] = "DZ";  varname[0] = "U";  varname[0] = "V";  varname[0] = "W"; varname[0] = "P"; 
   }
   
   // We have to fill these vectors from the json input file
   std::vector<std::string> facenamearray;
   std::vector<ParsedFunction> parsedfunctionarray;
   std::vector<BDCType> bdctypearray;

   // These information must be retrieved from multilevel_mesh
   unsigned int bdcsize = 0;
   
   // loop over the variables
   for(int ivar=0; ivar<numvar; ivar++) {
     std::stringstream ss;
     ss << "multilevel_solution.multilevel_mesh.first.variable." << varname[ivar] << ".boundary_conditions";
     bdcsize = inputparser->getSize(ss.str());
     for(unsigned int index=0; index<bdcsize; ++index) {
       //facename
       std::string facename = inputparser->getValueFromArray(ss.str(), index, "facename", "top");
       facenamearray.push_back(facename);
 
       //bdctype
       BDCType bdctype = inputparser->getValueFromArray(ss.str(), index, "bdc_type", DIRICHLET);
       bdctypearray.push_back(bdctype);
   
       //function
       std::string bdcfuncstr = inputparser->getValueFromArray(ss.str(), index, "bdc_func", "0.");
       ParsedFunction pfunc(bdcfuncstr, "x,y,z,t");
       parsedfunctionarray.push_back(pfunc);
    }
  }
   
   for(int ivar=0; ivar<numvar; ivar++) {
     for(unsigned int index2=0; index2<bdcsize; ++index2) {   
       ml_sol.SetBoundaryCondition_new(varname[ivar],facenamearray[index2+ivar*bdcsize],bdctypearray[index2+ivar*bdcsize],false,&parsedfunctionarray[index2+ivar*bdcsize]);
     }
   }
    
   ml_sol.GenerateBdc("All");
   
   
//   if(1==simulation || 2==simulation)
//     ml_sol.AttachSetBoundaryConditionFunction(SetBoundaryConditionTurek);
//   else if( 3==simulation)
//     ml_sol.AttachSetBoundaryConditionFunction(SetBoundaryConditionDrum);
//   else if (4==simulation || 6==simulation)
//     ml_sol.AttachSetBoundaryConditionFunction(SetBoundaryConditionBatheCylinder);
//   else if (5==simulation)
//     ml_sol.AttachSetBoundaryConditionFunction(SetBoundaryConditionBatheShell);
//   else if (7 == simulation)
//     ml_sol.AttachSetBoundaryConditionFunction(SetBoundaryConditionComsol);
// 
//   ml_sol.GenerateBdc("DX","Steady");
//   ml_sol.GenerateBdc("DY","Steady");
//   if (dimension == 3) 
//     ml_sol.GenerateBdc("DZ","Steady");
//   ml_sol.GenerateBdc("U","Steady");
//   ml_sol.GenerateBdc("V","Steady");
//   if (dimension == 3) 
//     ml_sol.GenerateBdc("W","Steady");
//   ml_sol.GenerateBdc("P","Steady");
  
  MultiLevelProblem ml_prob(&ml_sol);
  
  double rhof, muf, rhos, ni, E; 
  string solid_model, fluid_model;
  E = inputparser->getValue("multilevel_problem.parameters.solid.young_module", 1.);
  ni = inputparser->getValue("multilevel_problem.parameters.solid.poisson_coefficient", 0.);
  rhos = inputparser->getValue("multilevel_problem.parameters.solid.density", 1.);
  solid_model = inputparser->getValue("multilevel_problem.parameters.solid.model", "default_model");
  Solid solid(par,E,ni,rhos,solid_model.c_str()); ;
  cout << "Solid properties: " << endl;
  cout << solid << endl;
  
  // Generate Fluid Object
  muf = inputparser->getValue("multilevel_problem.parameters.fluid.dynamic_viscosity", 1.);
  rhof = inputparser->getValue("multilevel_problem.parameters.fluid.density", 1.);
  fluid_model = inputparser->getValue("multilevel_problem.parameters.fluid.model", "default_model");
  Fluid fluid(par,muf,rhof,fluid_model.c_str());
  cout << "Fluid properties: " << endl;
  cout << fluid << endl;
  
  // Add fluid object
  ml_prob.parameters.set<Fluid>("Fluid") = fluid;
  // Add Solid Object
  ml_prob.parameters.set<Solid>("Solid") = solid;
  // mark Solid nodes
  ml_msh.MarkStructureNode();
 
  //create systems
  // add the system FSI to the MultiLevel problem
  MonolithicFSINonLinearImplicitSystem & system = ml_prob.add_system<MonolithicFSINonLinearImplicitSystem> ("Fluid-Structure-Interaction");
  system.AddSolutionToSystemPDE("DX");
  system.AddSolutionToSystemPDE("DY");
  if (dimension == 3) 
    system.AddSolutionToSystemPDE("DZ");
  system.AddSolutionToSystemPDE("U");
  system.AddSolutionToSystemPDE("V");
  if (dimension == 3) 
    system.AddSolutionToSystemPDE("W");
  system.AddSolutionToSystemPDE("P");
  

  // System Fluid-Structure-Interaction
  system.SetAssembleFunction(IncompressibleFSIAssemblyAD_DD);  
  
  system.SetMgType(F_CYCLE);
  system.SetAbsoluteConvergenceTolerance(1.e-10);
  system.SetNumberPreSmoothingStep(1);
  system.SetNumberPostSmoothingStep(1);
  
  unsigned int max_number_linear_iteration = inputparser->getValue("multilevel_problem.multilevel_mesh.first.system.fsi.linear_solver.max_number_linear_iteration",6);
  system.SetMaxNumberOfLinearIterations(max_number_linear_iteration);
  
  unsigned int max_number_nonlinear_iteration = inputparser->getValue("multilevel_problem.multilevel_mesh.first.system.fsi.non_linear_solver.max_number_nonlinear_iteration",6);
  system.SetMaxNumberOfNonLinearIterations(max_number_nonlinear_iteration);
  
  double nonlinear_convergence_tolerance = inputparser->getValue("multilevel_problem.multilevel_mesh.first.system.fsi.non_linear_solver.abs_conv_tol",1.e-5);
  system.SetNonLinearConvergenceTolerance(nonlinear_convergence_tolerance);
  
  //Set Smoother Options
  if(inputparser->isTrue("multilevel_problem.multilevel_mesh.first.system.fsi.linear_solver.type.multigrid.smoother.type","gmres")) {
    system.SetMgSmoother(GMRES_SMOOTHER);
  }
  else if(inputparser->isTrue("multilevel_problem.multilevel_mesh.first.system.fsi.linear_solver.type.multigrid.smoother.type","asm")) {
    system.SetMgSmoother(ASM_SMOOTHER);
  }
  else if(inputparser->isTrue("multilevel_problem.multilevel_mesh.first.system.fsi.linear_solver.type.multigrid.smoother.type","vanka")) {
    system.SetMgSmoother(VANKA_SMOOTHER);
  }
  
  // init all the systems
  system.init();
  
  //Set Smoother-solver Options
  if(inputparser->isTrue("multilevel_problem.multilevel_mesh.first.system.fsi.linear_solver.type.multigrid.smoother.type","gmres")) {
    system.SetDirichletBCsHandling(ELIMINATION);  
    system.SetSolverFineGrids(GMRES);
    std::string precond = inputparser->getValue("multilevel_problem.multilevel_mesh.first.system.fsi.linear_solver.type.multigrid.smoother.type.gmres.precond","ilu");
    if(precond == "ilu") {
      system.SetPreconditionerFineGrids(ILU_PRECOND); 
    }
    else if(precond == "mlu") {
      system.SetPreconditionerFineGrids(MLU_PRECOND);
    }
    else {
     return 1; 
    }
  }
  else if(inputparser->isTrue("multilevel_problem.multilevel_mesh.first.system.fsi.linear_solver.type.multigrid.smoother.type","asm")) {
    system.SetSolverFineGrids(GMRES);
    system.SetDirichletBCsHandling(ELIMINATION);  
    std::string precond = inputparser->getValue("multilevel_problem.multilevel_mesh.first.system.fsi.linear_solver.type.multigrid.smoother.type.asm.precond","ilu");
    if(precond == "ilu") {
      system.SetPreconditionerFineGrids(ILU_PRECOND); 
    }
    else if(precond == "mlu") {
      system.SetPreconditionerFineGrids(MLU_PRECOND);
    }
    else {
     return 1; 
    }
  }
  else if(inputparser->isTrue("multilevel_problem.multilevel_mesh.first.system.fsi.linear_solver.type.multigrid.smoother.type","vanka")) {
    system.SetSolverFineGrids(GMRES);
    std::string precond = inputparser->getValue("multilevel_problem.multilevel_mesh.first.system.fsi.linear_solver.type.multigrid.smoother.type.vanka.precond","ilu");
    if(precond == "ilu") {
      system.SetPreconditionerFineGrids(ILU_PRECOND); 
    }
    else if(precond == "mlu") {
      system.SetPreconditionerFineGrids(MLU_PRECOND);
    }
    else {
     return 1; 
    }
  }
  
  
  system.SetTolerances(1.e-12,1.e-20,1.e+50,20);
 
  system.ClearVariablesToBeSolved();
  system.AddVariableToBeSolved("All");
  
  system.AddVariableToBeSolved("DX");
  system.AddVariableToBeSolved("DY");
  if (dimension == 3) 
    system.AddVariableToBeSolved("DZ");
  
  system.AddVariableToBeSolved("U");
  system.AddVariableToBeSolved("V");
  if (dimension == 3) 
    system.AddVariableToBeSolved("W");
  system.AddVariableToBeSolved("P");
    
  //for Vanka and ASM smoothers
  system.SetNumberOfSchurVariables(1);
  if(inputparser->isTrue("multilevel_problem.multilevel_mesh.first.system.fsi.linear_solver.type.multigrid.smoother.type","asm")) {
    if(inputparser->isTrue("multilevel_problem.multilevel_mesh.first.system.fsi.linear_solver.type.multigrid.smoother.type.asm","fsi")) {
      unsigned int blockfluidnumber = inputparser->getValue("multilevel_problem.multilevel_mesh.first.system.fsi.linear_solver.type.multigrid.smoother.type.asm.block_fluid_number",2);
      system.SetElementBlockNumberFluid(blockfluidnumber);
      std::string allsolidblock = inputparser->getValue("multilevel_problem.multilevel_mesh.first.system.fsi.linear_solver.type.multigrid.smoother.type.asm.block_solid_number","all");     
      if(allsolidblock == "all") {
        system.SetElementBlockSolidAll();
      }
    }
    else {
      unsigned int blocknumber = inputparser->getValue("multilevel_problem.multilevel_mesh.first.system.fsi.linear_solver.type.multigrid.smoother.type.asm.block_number",2);
      system.SetElementBlockNumber(blocknumber);
    }
  }
    
  ml_sol.SetWriter(VTK);

  std::vector<std::string> mov_vars;
  mov_vars.push_back("DX");
  mov_vars.push_back("DY");
  mov_vars.push_back("DZ");
  ml_sol.GetWriter()->SetMovingMesh(mov_vars);
  
  // Solving Fluid-Structure-Interaction system
  std::cout << std::endl;
  std::cout << " *********** Fluid-Structure-Interaction ************  " << std::endl;
  system.solve();
   
  //print solution 
  std::vector<std::string> print_vars;
  print_vars.push_back("DX");
  print_vars.push_back("DY");
  if (dimension == 3) 
    print_vars.push_back("DZ");
  print_vars.push_back("U");
  print_vars.push_back("V");
  if (dimension == 3) 
    print_vars.push_back("W");
  print_vars.push_back("P");
      
  ml_sol.GetWriter()->write(DEFAULT_OUTPUTDIR,"biquadratic",print_vars);

  // Destroy all the new systems
  ml_prob.clear();
   
  return 0;
}


bool SetRefinementFlag(const double &x, const double &y, const double &z, const int &elemgroupnumber,const int &level) {
  bool refine=0;

  //refinemenet based on elemen group number
  if (elemgroupnumber==5) refine=1;
  if (elemgroupnumber==6) refine=1;
  if (elemgroupnumber==7 && level<5) refine=1;

  return refine;

}

//---------------------------------------------------------------------------------------------------------------------

bool SetBoundaryConditionTurek(const double &x, const double &y, const double &z,const char name[], double &value, const int facename, const double time) {
  bool test=1; //dirichlet
  value=0.;
  if(!strcmp(name,"U")) {
    if(1==facename){   //inflow
      test=1;
      double um = 0.2;
      value=1.5*um*4.0/0.1681*y*(0.41-y);
    }  
    else if(2==facename ){  //outflow
     test=0;
 //    test=1;
     value=0.;
    }
    else if(3==facename ){  // no-slip fluid wall
      test=1;
      value=0.;	
    }
    else if(4==facename ){  // no-slip solid wall
      test=1;
      value=0.;
    }
    else if(6==facename ){   // beam case zero stress
      test=0;
      value=0.;
    }
  }  
  else if(!strcmp(name,"V")){
    if(1==facename){            //inflow
      test=1;
      value=0.;
    }  
    else if(2==facename ){      //outflow
     test=0;
 //    test=1;
     value=0.;
    }
    else if(3==facename ){      // no-slip fluid wall
      test=1;
      value=0;
    }
    else if(4==facename ){      // no-slip solid wall
      test=1;
      value=0.;
    }
    else if(6==facename ){   // beam case zero stress
      test=0;
      value=0.;
    }
  }
  else if(!strcmp(name,"P")){
    if(1==facename){
      test=0;
      value=0.;
    }  
    else if(2==facename ){  
      test=0;
      value=0.;
    }
    else if(3==facename ){  
      test=0;
      value=0.;
    }
    else if(4==facename ){  
      test=0;
      value=0.;
    }
    else if(6==facename ){   // beam case zero stress
      test=0;
      value=0.;
    }
  }
  else if(!strcmp(name,"DX")){
    if(1==facename){         //inflow
      test=1;
      value=0.;
    }  
    else if(2==facename ){   //outflow
     test=1;
     value=0.;
    }
    else if(3==facename ){   // no-slip fluid wall
      test=0; //0
      value=0.;	
    }
    else if(4==facename ){   // no-slip solid wall
      test=1;
      value=0.;
    }
    else if(6==facename ){   // beam case zero stress
      test=0;
      value=0.;
    }
  }
  else if(!strcmp(name,"DY")){
    if(1==facename){         //inflow
      test=0; // 0
      value=0.;
    }  
    else if(2==facename ){   //outflow
     test=0; // 0
     value=0.;
    }
    else if(3==facename ){   // no-slip fluid wall
      test=1;
      value=0.;	
    }
    else if(4==facename ){   // no-slip solid wall
      test=1;
      value=0.;
    }
    else if(6==facename ){   // beam case zero stress
      test=0;
      value=0.;
    }
  }
  return test;
}

bool SetBoundaryConditionDrum(const double &x, const double &y, const double &z,const char name[], double &value, const int facename, const double time) {
  bool test=1; //dirichlet
  value=0.;
  if(!strcmp(name,"U")) {
    if(1==facename){   //top
      test=0;
      value=0;
    }  
    else if(2==facename ){  //top side
     test=0;
     value=0.;
    }
    else if(3==facename ){  //top bottom
      test=1;
      value=0;	
    }
    else if(4==facename ){  //solid side
      test=1;
      value=0;	
    }
    else if(5==facename ){  //bottom side
      test=1;
      value=0;	
    }
    else if(6==facename ){  //bottom 
      test=0;
      value=200000;	
    }
  }  
  else if(!strcmp(name,"V")){
     if(1==facename){   //top
      test=0;
      value=0;
    }  
    else if(2==facename ){  //top side
     test=0;
     value=0.;
    }
    else if(3==facename ){  //top bottom
      test=1;
      value=0;	
    }
    else if(4==facename ){  //solid side
      test=1;
      value=0;	
    }
    else if(5==facename ){  //bottom side
      test=1;
      value=0;	
    }
    else if(6==facename ){  //bottom 
      test=0;
      value=0;	
    }
  }
  else if(!strcmp(name,"P")){
    if(facename==facename){
      test=0;
      value=0.;
    } 
  }
  else if(!strcmp(name,"DX")){
    if(1==facename){   //top
      test=0;
      value=0;
    }  
    else if(2==facename ){  //top side
     test=1;
     value=0.;
    }
    else if(3==facename ){  //top bottom
      test=0;
      value=0;	
    }
    else if(4==facename ){  //solid side
      test=1;
      value=0;	
    }
    else if(5==facename ){  //bottom side
      test=1;
      value=0;	
    }
    else if(6==facename ){  //bottom 
      test=0;
      value=0;	
    }
  }
  else if(!strcmp(name,"DY")){
   if(1==facename){   //top
      test=1;
      value=0;
    }  
    else if(2==facename ){  //top side
     test=0;
     value=0.;
    }
    else if(3==facename ){  //top bottom
      test=1;
      value=0;	
    }
    else if(4==facename ){  //solid side
      test=1;
      value=0;	
    }
    else if(5==facename ){  //bottom side
      test=1;
      value=0;	
    }
    else if(6==facename ){  //bottom 
      test=1;
      value=0;	
    }
  }

  return test;
}



bool SetBoundaryConditionBatheCylinder(const double &x, const double &y, const double &z,const char name[], double &value, const int facename, const double time) {
  bool test=1; //dirichlet
  value=0.;
  
  if(!strcmp(name,"U")) {
    if(1==facename){   //inflow
      //test=1;
      //double r=sqrt(y*y+z*z);
      //value=1000*(0.05-r)*(0.05+r);
      test=0;
      value=15*1.5*1000;
    }  
    else if(2==facename){  //outflow
      test=0;
      value=13*1.5*1000;   
    }
    else if(3==facename || 4==facename ){  // clamped solid 
      test=1;
      value=0.;
    } 
    else if(5==facename ){  // free solid
      test=0;
      value=0.;
    } 
  }  
  else if(!strcmp(name,"V")){
    if(1==facename){   //inflow
      test=0;
      value=0;
    }  
    else if(2==facename){  //outflow
      test=0;
      value=0;
    }
    else if(3==facename || 4==facename ){  // clamped solid 
      test=1;
      value=0.;
    } 
    else if(5==facename ){  // free solid
      test=0;
      value=0.;
    } 
  }
  else if(!strcmp(name,"W")){
    if(1==facename){   //inflow
      test=0;
      value=0;
    }  
    else if(2==facename){  //outflow
      test=0;
      value=0;
    }
    else if(3==facename || 4==facename ){  // clamped solid 
      test=1;
      value=0.;
    } 
    else if(5==facename ){  // free solid
      test=0;
      value=0.;
    } 
  }
  else if(!strcmp(name,"P")){
    if(1==facename){   //outflow
      test=0;
      value=0;
    }  
    else if(2==facename){  //inflow
      test=0;
      value=0;
    }
    else if(3==facename || 4==facename ){  // clamped solid 
      test=0;
      value=0.;
    } 
    else if(5==facename ){  // free solid
      test=0;
      value=0.;
    } 
  }
  else if(!strcmp(name,"DX")){
   if(1==facename){   //outflow
      test=1;
      value=0;
    }  
    else if(2==facename){  //inflow
      test=1;
      value=0;
    }
    else if(3==facename || 4==facename ){  // clamped solid 
      test=1;
      value=0.;
    } 
    else if(5==facename ){  // free solid
      test=0;
      value=0.;
    } 
  }
  else if(!strcmp(name,"DY")){
    if(1==facename){   //outflow
      test=0;
      value=0;
    }  
    else if(2==facename){  //inflow
      test=0;
      value=0;
    }
    else if(3==facename || 4==facename ){  // clamped solid 
      test=1;
      value=0.;
    } 
    else if(5==facename ){  // free solid
      test=0;
      value=0.;
    } 
  }
  else if(!strcmp(name,"DZ")){
    if(1==facename){   //outflow
      test=0;
      value=0;
    }  
    else if(2==facename){  //inflow
      test=0;
      value=0;
    }
    else if(3==facename || 4==facename ){  // clamped solid 
      test=1;
      value=0.;
    } 
    else if(5==facename ){  // free solid
      test=0;
      value=0.;
    } 
  }

  return test;
}

bool SetBoundaryConditionBatheShell(const double &x, const double &y, const double &z,const char name[], double &value, const int facename, const double time) {
  bool test=1; //dirichlet
  value=0.;
  
  if(!strcmp(name,"U")) {
    if(2==facename){  //stress
      test=0;
      value=15*1.5*1000;   
    }
    else if(3==facename || 4==facename ){  // clamped solid 
      test=1;
      value=0.;
    } 
    else if(5==facename ){  // free solid
      test=0;
      value=0.;
    } 
  }  
  else if(!strcmp(name,"V")){
    if(2==facename){  //stress
      test=0;
      value=0;
    }
    else if(3==facename || 4==facename ){  // clamped solid 
      test=1;
      value=0.;
    } 
    else if(5==facename ){  // free solid
      test=0;
      value=0.;
    } 
  }
  else if(!strcmp(name,"W")){
    if(2==facename){  //stress
      test=0;
      value=0;
    }
    else if(3==facename || 4==facename ){  // clamped solid 
      test=1;
      value=0.;
    } 
    else if(5==facename ){  // free solid
      test=0;
      value=0.;
    } 
  }
  else if(!strcmp(name,"P")){
    if(2==facename){  //stress
      test=0;
      value=0;
    }
    else if(3==facename || 4==facename ){  // clamped solid 
      test=0;
      value=0.;
    } 
    else if(5==facename ){  // free solid
      test=0;
      value=0.;
    } 
  }
  else if(!strcmp(name,"DX")){
   if(2==facename){  //stress
      test=0;
      value=0;
    }
    else if(3==facename || 4==facename ){  // clamped solid 
      test=1;
      value=0.;
    } 
    else if(5==facename ){  // free solid
      test=0;
      value=0.;
    } 
  }
  else if(!strcmp(name,"DY")){
    if(2==facename){  //stress
      test=0;
      value=0;
    }
    else if(3==facename || 4==facename ){  // clamped solid 
      test=1;
      value=0.;
    } 
    else if(5==facename ){  // free solid
      test=0;
      value=0.;
    } 
  }
  else if(!strcmp(name,"DZ")){
    if(2==facename){  //stress
      test=0;
      value=0;
    }
    else if(3==facename || 4==facename ){  // clamped solid 
      test=1;
      value=0.;
    } 
    else if(5==facename ){  // free solid
      test=0;
      value=0.;
    } 
  }

  return test;
}



//---------------------------------------------------------------------------------------------------------------------

bool SetBoundaryConditionComsol(const double &x, const double &y, const double &z,const char name[], double &value, const int FaceName, const double time) {
  bool test=1; //Dirichlet
  value=0.;
  //   cout << "Time bdc : " <<  time << endl;
  if (!strcmp(name,"U")) {
    if (1==FaceName) { //inflow
      test=1;
      //comsol Benchmark
      //value = (0.05*time*time)/(sqrt( (0.04 - time*time)*(0.04 - time*time) + (0.1*time)*(0.1*time) ))*y*(0.0001-y)*4.*100000000;
      value = 0.05*y*(0.0001-y)*4.*100000000;
    }
    else if (2==FaceName ) {  //outflow
      test=0;
      //    test=1;
      value=0.;
    }
    else if (3==FaceName ) {  // no-slip fluid wall
      test=1;
      value=0.;
    }
    else if (4==FaceName ) {  // no-slip solid wall
      test=1;
      value=0.;
    }
  }
  else if (!strcmp(name,"V")) {
    if (1==FaceName) {          //inflow
      test=1;
      value=0.;
    }
    else if (2==FaceName ) {    //outflow
      test=0;
      //    test=1;
      value=0.;
    }
    else if (3==FaceName ) {    // no-slip fluid wall
      test=1;
      value=0;
    }
    else if (4==FaceName ) {    // no-slip solid wall
      test=1;
      value=0.;
    }
  }
  else if (!strcmp(name,"W")) {
    if (1==FaceName) {
      test=1;
      value=0.;
    }
    else if (2==FaceName ) {
      test=1;
      value=0.;
    }
    else if (3==FaceName ) {
      test=1;
      value=0.;
    }
    else if (4==FaceName ) {
      test=1;
      value=0.;
    }
  }
  else if (!strcmp(name,"P")) {
    if (1==FaceName) {
      test=0;
      value=0.;
    }
    else if (2==FaceName ) {
      test=0;
      value=0.;
    }
    else if (3==FaceName ) {
      test=0;
      value=0.;
    }
    else if (4==FaceName ) {
      test=0;
      value=0.;
    }
  }
  else if (!strcmp(name,"DX")) {
    if (1==FaceName) {       //inflow
      test=1;
      value=0.;
    }
    else if (2==FaceName ) { //outflow
      test=1;
      value=0.;
    }
    else if (3==FaceName ) { // no-slip Top fluid wall
      test=0;
      value=0;
    }
    else if (4==FaceName ) { // no-slip solid wall
      test=1;
      value=0.;
    }
  }
  else if (!strcmp(name,"DY")) {
    if (1==FaceName) {       //inflow
      test=0;
      value=0.;
    }
    else if (2==FaceName ) { //outflow
      test=0;
      value=0.;
    }
    else if (3==FaceName ) { // no-slip fluid wall
      test=1;
      value=0.;
    }
    else if (4==FaceName ) { // no-slip solid wall
      test=1;
      value=0.;
    }
  }
  else if (!strcmp(name,"DZ")) {
    if (1==FaceName) {       //inflow
      test=1;
      value=0.;
    }
    else if (2==FaceName ) { //outflow
      test=1;
      value=0.;
    }
    else if (3==FaceName ) { // no-slip fluid wall
      test=1;
      value=0.;
    }
    else if (4==FaceName ) { // no-slip solid wall
      test=1;
      value=0.;
    }
  }
  else if (!strcmp(name,"AX")) {
    test=0;
    value=0;
  }
  else if (!strcmp(name,"AY")) {
    test=0;
    value=0;
  }
  else if (!strcmp(name,"AZ")) {
    test=0;
    value=0;
  }
  return test;
}

