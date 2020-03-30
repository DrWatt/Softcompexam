#define DTTree_cxx
#include "DTTree.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <iostream>
#include <fstream>
/*int main()
{
	DTTree m;
	m.Loop();
}*/
void DTTree::Loop()
{
	ofstream o;
	o.open("datatree2.csv");
	o << "bxout,bx,phi,phiB,wheel,sector,station,quality,is2nd\n";

   if (fChain == 0) return;

   Long64_t nentries = fChain->GetEntriesFast();

   Long64_t nbytes = 0, nb = 0;
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      // if (Cut(ientry) < 0) continue;
      int ov = 0;
      float t = jentry/1000.;
      if (t == (int)t) clog <<'\r' << "Entry " << jentry;
      for (unsigned int i = 0; i < ltTwinMuxIn_phi->size(); ++i)
      {
     		for (int j = 0; j < ltTwinMuxOut_bx->size(); ++j)
     		{
     			if(ov==0&&ltTwinMuxIn_wheel->at(i) == ltTwinMuxOut_wheel->at(j) && ltTwinMuxIn_sector->at(i) == ltTwinMuxOut_sector->at(j)&&ltTwinMuxIn_station->at(i)==ltTwinMuxOut_station->at(j))
     			{

     				o << ltTwinMuxOut_bx->at(j) << ',';
     				o << ltTwinMuxIn_bx->at(i) << ',';
            		o << ltTwinMuxIn_phi->at(i) << ',';
            		o << ltTwinMuxIn_phiB->at(i)<< ',';
            		o << ltTwinMuxIn_wheel->at(i)<< ',';
            		o << ltTwinMuxIn_sector->at(i)<< ',';
            		o << ltTwinMuxIn_station->at(i)<< ',';
					o << ltTwinMuxIn_quality->at(i)<< ',';
					o << ltTwinMuxIn_is2nd->at(i);
					o << '\n';
					++ov;

     			}
     		}
        ov=0;

      }



   }
   o.close();
}
