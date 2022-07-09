/*
 * JCuda - Java bindings for CUDA
 *
 * http://www.jcuda.org
 */
package jcuda.jcudnn;


import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.List;

import org.junit.Test;

/**
 * Basic test of the bindings of the JCudnn class
 */
public class JCudnnBasicBindingTest
{
    private static final List<String> SKIPPED = Arrays.asList(
        "cudnnCTCLoss_v8"
    );

    public static void main(String[] args)
    {
        BasicBindingTest.testBinding(JCudnn.class, SKIPPED);
    }

    @Test
    public void testJCudnn()
    {
        assertTrue(BasicBindingTest.testBinding(JCudnn.class, SKIPPED));
    }
    

}
