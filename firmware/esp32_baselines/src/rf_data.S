.section .rodata.rf_model,"a"
.balign 4
.global rf_model_start
rf_model_start:
.incbin "include/rf_forest.bin"
.global rf_model_end
rf_model_end:
