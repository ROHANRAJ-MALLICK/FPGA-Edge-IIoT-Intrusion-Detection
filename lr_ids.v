`timescale 1ns / 1ps

module lr_ids_core (
    input wire clk,
    input wire rst_n,

    // AXI-Stream Input (Features)
    input wire [7:0] s_axis_tdata,
    input wire s_axis_tvalid,
    output reg s_axis_tready,
    input wire s_axis_tlast,

    // AXI-Stream Output (Classification Label)
    output reg [7:0] m_axis_tdata, // Padded to 8-bit, LSB is the label
    output reg m_axis_tvalid,
    input wire m_axis_tready,
    output reg m_axis_tlast,

    // Configuration Inputs (Driven by AXI-Lite wrapper)
    input wire signed [15:0] threshold, 
    input wire [61*32-1:0] flat_weights, // 61 weights in 32-bit Q16.16 format
    input wire signed [31:0] bias
);

    // Internal Registers for streaming in the 61 features
    reg [7:0] feature_regs [0:60];
    reg [5:0] feature_count;
    
    // DSP Array and Accumulator signals
    wire signed [31:0] weights [0:60];
    wire signed [39:0] mult_results [0:60]; 
    reg signed [47:0] accumulator;          
    
    // FIX: Variables declared at the module level
    integer j;
    reg signed [47:0] comb_sum;

    // Unpack flattened weights from AXI-Lite
    genvar i;
    generate
        for (i = 0; i < 61; i = i + 1) begin : unpack_weights
            assign weights[i] = flat_weights[(i*32) +: 32];
        end
    endgenerate

    // Multiplier Array (Maps strictly to the 61 DSPs as per documentation)
    generate
        for (i = 0; i < 61; i = i + 1) begin : dsp_array
            // Convert 8-bit unsigned feature to signed, then multiply by signed Q16.16 weight
            assign mult_results[i] = $signed({1'b0, feature_regs[i]}) * weights[i];
        end
    endgenerate

    // FIX: Combinational Adder Tree
    always @(*) begin
        comb_sum = bias;
        for (j = 0; j < 61; j = j + 1) begin
            comb_sum = comb_sum + mult_results[j];
        end
    end

    // State Machine logic for shifting features and computing
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            feature_count <= 0;
            s_axis_tready <= 1;
            m_axis_tvalid <= 0;
            accumulator <= 0;
        end else begin
            // Shift-in phase
            if (s_axis_tvalid && s_axis_tready) begin
                feature_regs[feature_count] <= s_axis_tdata;
                
                if (feature_count == 60) begin
                    // We have all 61 features, trigger computation next cycle
                    s_axis_tready <= 0; 
                    feature_count <= 0;
                end else begin
                    feature_count <= feature_count + 1;
                end
            end

            // Computation and Output Phase
            if (!s_axis_tready && !m_axis_tvalid) begin
                
                accumulator <= comb_sum; // Register the final sum
                
                // Threshold comparison: label1 = 1 if y >= theta1
                // We extract [31:16] because the fractional part of Q16.16 takes up the bottom 16 bits.
                if ($signed(comb_sum[31:16]) >= threshold) begin
                    m_axis_tdata <= 8'h01; // Attack
                end else begin
                    m_axis_tdata <= 8'h00; // Normal
                end
                
                m_axis_tvalid <= 1;
                m_axis_tlast <= 1;
            end

            // Handshake output
            if (m_axis_tvalid && m_axis_tready) begin
                m_axis_tvalid <= 0;
                s_axis_tready <= 1; // Ready for next sample
            end
        end
    end
endmodule