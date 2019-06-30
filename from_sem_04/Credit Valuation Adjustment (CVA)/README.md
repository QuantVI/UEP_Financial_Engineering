######The project
##Project scope: Credit Valuation Adjustments for Derivative Contracts###### Key task: Prepare a report describing applied methodology, as well as CVA calculation results for different types of derivative instruments.

###Analyzed derivatives:##### • Buy EUR sell PLN FX Forward (maturity: 1 year, notional: 100 000 EUR; strike: 4,3930)
##### • Receiver (we receive fixed rate) PLN WIBOR 3M IRS (maturity: 3 years, notional: 500 000 PLN; fixed rate: 2,2144% - paid annually)
##### • EUR/PLN CIRS, paid rate: fixed rate (-0,0575%) in EUR, received rate WIBOR 3M, notional exchange at the beginning as well as end of the contract (maturity: 3 years, notional: 100 000 EUR/ 430 000 PLN)

--

#### I. Analysis of derivative instruments in question:• Mechanics,  • Payoff formula/ payoff profile,  
• Valuation formula.#### II. Implied default probability:• CDS contracts: mechanics,  • CDS contracts: valuation,  • Implying probability of default from a CDS spread: CDS bootstrapping,  • CDS bootstrapping for sample market quotes  
(IMPORTANT: algorithm implemented in code form)

#### III. Risk factors simulation:• Proposition of simulation methodology,  • Calibration to market data,  • Simulation: 3 years horizon, 1 month time step, minimum number of simulations 1 000.  #### IV. Calculation of expected exposure profiles (EE) for derivatives in question:• Presentation and identification of key features of EE for the analyzed derivatives,  • Sensitivity analysis – describe the impact of increased volatility on EE of the analyzed derivatives.

#### V. CVA calculation for derivatives in question :• CVA calculation for the analyzed derivatives,  • Sensitivity analysis – describe the impact of increased volatility on CVA of the analyzed derivatives.
