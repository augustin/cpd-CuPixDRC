CuPixDRC
============

VLSI pixel-based design rule checker in CUDA. It uses the text-based rendered output of [`ChipLib`](https://github.com/waddlesplash/cpd-ChipLib) as its input.

I don't plan on maintaining this, but feel free to use it under the MIT license. Make sure to read the caveats below.

What's this?
--------------------------------
My science fair project from 2013-2014. Yes, it's a bit advanced for a science fair. It won first place my local fair but didn't win anything at the regional fair (I think the judges there didn't understand my project... :/). As far as I can tell, this project was pretty original -- I only found one other CUDA-based design-rule-checking project, and it was polygon-based instead of pixel-based (which I think is suboptimal for GPUs) and was a research project done by some grad students. I did this in 11th grade.

The below caveats give a general overview of the code. For a general overview of the project and the methodologies and general implementation concepts, you probably want to read [the paper](https://github.com/waddlesplash/cpd-CuPixDRC/blob/master/docs/paper.pdf?raw=true).

The code is a bit of a mess, and it requires a checkout of [`ChipLib`](https://github.com/waddlesplash/cpd-ChipLib) in the same directory that this is checked out in (unless I removed that dependency -- at any rate, this uses the .txt files that `ChipDisplay` from `ChipLib` generates, so this is pretty useless without that).

The CUDA kernel itself can run in both CUDA mode or CPU mode -- it's compiled twice so you can run it as either. If the code for the kernel looks wacky, that's because it was optimized based on the PTX assembly that the CUDA compiler spit out -- from the looks of it, either the LLVM optimizer or the PTX one could use a bunch of work.

My only tests of this were on Kubuntu 14.04 with a NVIDIA GeForce GTX 550 Ti (1GB version) and the latest-at-the-time proprietary drivers. I used a MIPS chip as my testcase, from a student project by Harvey Mudd College [HMC] (the original site has since gone off the web, but it looks like the Wayback Machine still has the page and at least some of the files [here](http://web.archive.org/web/20090425234903/http://www4.hmc.edu:8001/Engineering/158/07/project). You'll want the `chip.cif` file for use with this design rule checker.)

The code is a bit hacky, and can only check 2 rules: horizontal spacing, and vertical spacing. I tried to go with the same limitations specified in the latest-at-the-time MOSIS rules, which is what the HMC MIPS chip was supposed to conform to. The design rules are hardcoded to 1/14th scale of the CMF layer of the chip (as a full-size rendering was far too big to fit in the 1GB of GPU memory I had, and I don't have chunked-checking implemented), and that's defined in [`design_rules.h`](https://github.com/waddlesplash/cpd-CuPixDRC/blob/master/kernel/design_rules.h#L4) -- `ChipDisplay`'s export feature *should* be hardcoded to the same number but I didn't double-check when writing this.

If you have any questions, feel free to open an issue and ask away.
