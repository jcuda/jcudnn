/*
 * JCuda - Java bindings for CUDA
 *
 * http://www.jcuda.org
 */
package jcuda.jcudnn;


import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * Basic test of the bindings of the JCudnn class
 */
public class JCudnnBasicBindingTest
{
    public static void main(String[] args)
    {
        BasicBindingTest.testBinding(JCudnn.class);
    }

    @Test
    public void testJCudnn()
    {
        assertTrue(BasicBindingTest.testBinding(JCudnn.class));
    }
    

}
