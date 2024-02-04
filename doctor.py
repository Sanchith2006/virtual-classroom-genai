import viz
import vizfx
import vizinfo
import vizact
import viztask

viz.MainView.getHeadLight().disable()
sky_light = viz.addDirectionalLight(euler=(0,20,0))
sky_light.color(viz.WHITE)
sky_light.ambient([0.8]*3)
viz.setOption('viz.lightModel.ambient',[0]*3)
TRIAL_COUNT = 5				
TRIAL_DURATION = 20	
env=vizfx.addChild('D:\\details\\sensei.osgb')

viz.go()

