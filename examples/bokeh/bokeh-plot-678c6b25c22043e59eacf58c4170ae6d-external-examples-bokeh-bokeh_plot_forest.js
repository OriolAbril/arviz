(function() {
  var fn = function() {
    
    (function(root) {
      function now() {
        return new Date();
      }
    
      var force = false;
    
      if (typeof root._bokeh_onload_callbacks === "undefined" || force === true) {
        root._bokeh_onload_callbacks = [];
        root._bokeh_is_loading = undefined;
      }
    
      
      
    
      var element = document.getElementById("068cb474-8c96-4e3e-80e7-288b090ad2d9");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '068cb474-8c96-4e3e-80e7-288b090ad2d9' but no matching script tag was found.")
        }
      
    
      function run_callbacks() {
        try {
          root._bokeh_onload_callbacks.forEach(function(callback) {
            if (callback != null)
              callback();
          });
        } finally {
          delete root._bokeh_onload_callbacks
        }
        console.debug("Bokeh: all callbacks have finished");
      }
    
      function load_libs(css_urls, js_urls, callback) {
        if (css_urls == null) css_urls = [];
        if (js_urls == null) js_urls = [];
    
        root._bokeh_onload_callbacks.push(callback);
        if (root._bokeh_is_loading > 0) {
          console.debug("Bokeh: BokehJS is being loaded, scheduling callback at", now());
          return null;
        }
        if (js_urls == null || js_urls.length === 0) {
          run_callbacks();
          return null;
        }
        console.debug("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
        root._bokeh_is_loading = css_urls.length + js_urls.length;
    
        function on_load() {
          root._bokeh_is_loading--;
          if (root._bokeh_is_loading === 0) {
            console.debug("Bokeh: all BokehJS libraries/stylesheets loaded");
            run_callbacks()
          }
        }
    
        function on_error() {
          console.error("failed to load " + url);
        }
    
        for (var i = 0; i < css_urls.length; i++) {
          var url = css_urls[i];
          const element = document.createElement("link");
          element.onload = on_load;
          element.onerror = on_error;
          element.rel = "stylesheet";
          element.type = "text/css";
          element.href = url;
          console.debug("Bokeh: injecting link tag for BokehJS stylesheet: ", url);
          document.body.appendChild(element);
        }
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js": "qkRvDQVAIfzsJo40iRBbxt6sttt0hv4lh74DG7OK4MCHv4C5oohXYoHUM5W11uqS", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js": "Sb7Mr06a9TNlet/GEBeKaf5xH3eb6AlCzwjtU82wNPyDrnfoiVl26qnvlKjmcAd+", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js": "HaJ15vgfmcfRtB4c4YBOI4f1MUujukqInOWVqZJZZGK7Q+ivud0OKGSTn/Vm2iso"};
    
        for (var i = 0; i < js_urls.length; i++) {
          var url = js_urls[i];
          var element = document.createElement('script');
          element.onload = on_load;
          element.onerror = on_error;
          element.async = false;
          element.src = url;
          if (url in hashes) {
            element.crossOrigin = "anonymous";
            element.integrity = "sha384-" + hashes[url];
          }
          console.debug("Bokeh: injecting script tag for BokehJS library: ", url);
          document.head.appendChild(element);
        }
      };
    
      function inject_raw_css(css) {
        const element = document.createElement("style");
        element.appendChild(document.createTextNode(css));
        document.body.appendChild(element);
      }
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js"];
      var css_urls = [];
      
    
      var inline_js = [
        function(Bokeh) {
          Bokeh.set_log_level("info");
        },
        
        function(Bokeh) {
          (function() {
            var fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {
                    
                  var docs_json = '{"9ec5bc75-8df5-480c-b20a-49d19d69bf8e":{"roots":{"references":[{"attributes":{"source":{"id":"4637"}},"id":"4641","type":"CDSView"},{"attributes":{},"id":"4709","type":"Selection"},{"attributes":{"line_color":"#1f77b4","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"4673","type":"Line"},{"attributes":{},"id":"4747","type":"Selection"},{"attributes":{"source":{"id":"4627"}},"id":"4631","type":"CDSView"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":3.374380781729923},"y":{"value":1.65}},"id":"4649","type":"Circle"},{"attributes":{"data_source":{"id":"4622"},"glyph":{"id":"4623"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4624"},"selection_glyph":null,"view":{"id":"4626"}},"id":"4625","type":"GlyphRenderer"},{"attributes":{"source":{"id":"4612"}},"id":"4616","type":"CDSView"},{"attributes":{"data":{},"selected":{"id":"4737"},"selection_policy":{"id":"4738"}},"id":"4647","type":"ColumnDataSource"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.74129666559688},"y":{"value":1.95}},"id":"4663","type":"Circle"},{"attributes":{},"id":"4711","type":"Selection"},{"attributes":{},"id":"4713","type":"Selection"},{"attributes":{"source":{"id":"4687"}},"id":"4691","type":"CDSView"},{"attributes":{"data":{},"selected":{"id":"4719"},"selection_policy":{"id":"4720"}},"id":"4602","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"4652"},"glyph":{"id":"4653"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4654"},"selection_glyph":null,"view":{"id":"4656"}},"id":"4655","type":"GlyphRenderer"},{"attributes":{"source":{"id":"4597"}},"id":"4601","type":"CDSView"},{"attributes":{"data_source":{"id":"4577"},"glyph":{"id":"4578"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4579"},"selection_glyph":null,"view":{"id":"4581"}},"id":"4580","type":"GlyphRenderer"},{"attributes":{"source":{"id":"4622"}},"id":"4626","type":"CDSView"},{"attributes":{"data":{"x":[0.9649316231388204,5.949680273009057],"y":[1.65,1.65]},"selected":{"id":"4735"},"selection_policy":{"id":"4736"}},"id":"4642","type":"ColumnDataSource"},{"attributes":{},"id":"4704","type":"BasicTickFormatter"},{"attributes":{},"id":"4716","type":"UnionRenderers"},{"attributes":{},"id":"4739","type":"Selection"},{"attributes":{"data_source":{"id":"4662"},"glyph":{"id":"4663"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4664"},"selection_glyph":null,"view":{"id":"4666"}},"id":"4665","type":"GlyphRenderer"},{"attributes":{"source":{"id":"4602"}},"id":"4606","type":"CDSView"},{"attributes":{"line_alpha":0.1,"line_color":"#ff7f0e","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"4629","type":"Line"},{"attributes":{"source":{"id":"4662"}},"id":"4666","type":"CDSView"},{"attributes":{"toolbar":{"id":"4759"},"toolbar_location":"above"},"id":"4760","type":"ToolbarBox"},{"attributes":{"data":{},"selected":{"id":"4749"},"selection_policy":{"id":"4750"}},"id":"4677","type":"ColumnDataSource"},{"attributes":{},"id":"4730","type":"UnionRenderers"},{"attributes":{"toolbars":[{"id":"4568"}],"tools":[{"id":"4558"},{"id":"4559"},{"id":"4560"},{"id":"4561"},{"id":"4562"},{"id":"4563"},{"id":"4564"},{"id":"4565"}]},"id":"4759","type":"ProxyToolbar"},{"attributes":{},"id":"4750","type":"UnionRenderers"},{"attributes":{"ticks":[0.44999999999999996,2.0999999999999996]},"id":"4701","type":"FixedTicker"},{"attributes":{"line_color":"#1f77b4","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"4688","type":"Line"},{"attributes":{},"id":"4749","type":"Selection"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#ff7f0e"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.3706630373472235},"y":{"value":0.8999999999999999}},"id":"4634","type":"Circle"},{"attributes":{"line_alpha":0.1,"line_color":"#1f77b4","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"4654","type":"Line"},{"attributes":{"data":{},"selected":{"id":"4725"},"selection_policy":{"id":"4726"}},"id":"4617","type":"ColumnDataSource"},{"attributes":{"line_color":"#ff7f0e","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"4628","type":"Line"},{"attributes":{"bounds":"auto","end":3.4499999999999997,"min_interval":2,"start":-0.8999999999999999},"id":"4700","type":"DataRange1d"},{"attributes":{},"id":"4718","type":"UnionRenderers"},{"attributes":{},"id":"4743","type":"Selection"},{"attributes":{"data":{},"selected":{"id":"4755"},"selection_policy":{"id":"4756"}},"id":"4692","type":"ColumnDataSource"},{"attributes":{},"id":"4715","type":"Selection"},{"attributes":{"data_source":{"id":"4682"},"glyph":{"id":"4683"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4684"},"selection_glyph":null,"view":{"id":"4686"}},"id":"4685","type":"GlyphRenderer"},{"attributes":{"line_color":"#1f77b4","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"4638","type":"Line"},{"attributes":{"line_color":"#1f77b4","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"4658","type":"Line"},{"attributes":{"data":{"x":[1.9790083397469873,5.455951625836456],"y":[2.55,2.55]},"selected":{"id":"4753"},"selection_policy":{"id":"4754"}},"id":"4687","type":"ColumnDataSource"},{"attributes":{"data":{"x":[2.2921199846409115,6.479187946875487],"y":[0.8999999999999999,0.8999999999999999]},"selected":{"id":"4729"},"selection_policy":{"id":"4730"}},"id":"4627","type":"ColumnDataSource"},{"attributes":{},"id":"4733","type":"Selection"},{"attributes":{"data_source":{"id":"4637"},"glyph":{"id":"4638"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4639"},"selection_glyph":null,"view":{"id":"4641"}},"id":"4640","type":"GlyphRenderer"},{"attributes":{},"id":"4710","type":"UnionRenderers"},{"attributes":{},"id":"4745","type":"Selection"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":3.374380781729923},"y":{"value":1.65}},"id":"4648","type":"Circle"},{"attributes":{},"id":"4756","type":"UnionRenderers"},{"attributes":{"source":{"id":"4667"}},"id":"4671","type":"CDSView"},{"attributes":{"data_source":{"id":"4602"},"glyph":{"id":"4603"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4604"},"selection_glyph":null,"view":{"id":"4606"}},"id":"4605","type":"GlyphRenderer"},{"attributes":{"source":{"id":"4617"}},"id":"4621","type":"CDSView"},{"attributes":{"data":{"x":[1.9127415870254314,7.0411499662839026],"y":[2.25,2.25]},"selected":{"id":"4747"},"selection_policy":{"id":"4748"}},"id":"4672","type":"ColumnDataSource"},{"attributes":{"source":{"id":"4652"}},"id":"4656","type":"CDSView"},{"attributes":{},"id":"4712","type":"UnionRenderers"},{"attributes":{"line_alpha":0.1,"line_color":"#1f77b4","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"4674","type":"Line"},{"attributes":{"source":{"id":"4672"}},"id":"4676","type":"CDSView"},{"attributes":{"line_alpha":0.1,"line_color":"#ff7f0e","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"4614","type":"Line"},{"attributes":{},"id":"4738","type":"UnionRenderers"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.74129666559688},"y":{"value":1.95}},"id":"4664","type":"Circle"},{"attributes":{},"id":"4753","type":"Selection"},{"attributes":{"line_color":"#1f77b4","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"4653","type":"Line"},{"attributes":{"data":{"x":[-0.7842313478998125,9.985046696913068],"y":[2.55,2.55]},"selected":{"id":"4751"},"selection_policy":{"id":"4752"}},"id":"4682","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"4647"},"glyph":{"id":"4648"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4649"},"selection_glyph":null,"view":{"id":"4651"}},"id":"4650","type":"GlyphRenderer"},{"attributes":{"line_alpha":0.1,"line_color":"#ff7f0e","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"4609","type":"Line"},{"attributes":{"data_source":{"id":"4667"},"glyph":{"id":"4668"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4669"},"selection_glyph":null,"view":{"id":"4671"}},"id":"4670","type":"GlyphRenderer"},{"attributes":{},"id":"4729","type":"Selection"},{"attributes":{},"id":"4748","type":"UnionRenderers"},{"attributes":{"fill_color":{"value":"#ff7f0e"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.59253463805363},"y":{"value":0.6}},"id":"4618","type":"Circle"},{"attributes":{},"id":"4742","type":"UnionRenderers"},{"attributes":{"data":{},"selected":{"id":"4731"},"selection_policy":{"id":"4732"}},"id":"4632","type":"ColumnDataSource"},{"attributes":{"line_color":"#1f77b4","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"4668","type":"Line"},{"attributes":{"data_source":{"id":"4687"},"glyph":{"id":"4688"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4689"},"selection_glyph":null,"view":{"id":"4691"}},"id":"4690","type":"GlyphRenderer"},{"attributes":{"line_color":"#1f77b4","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"4643","type":"Line"},{"attributes":{"fill_color":{"value":"#ff7f0e"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.3706630373472235},"y":{"value":0.8999999999999999}},"id":"4633","type":"Circle"},{"attributes":{"data_source":{"id":"4657"},"glyph":{"id":"4658"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4659"},"selection_glyph":null,"view":{"id":"4661"}},"id":"4660","type":"GlyphRenderer"},{"attributes":{"data":{"x":[2.1521098502407368,6.9643589964055215],"y":[0.6,0.6]},"selected":{"id":"4723"},"selection_policy":{"id":"4724"}},"id":"4612","type":"ColumnDataSource"},{"attributes":{"source":{"id":"4607"}},"id":"4611","type":"CDSView"},{"attributes":{"data":{},"selected":{"id":"4743"},"selection_policy":{"id":"4744"}},"id":"4662","type":"ColumnDataSource"},{"attributes":{"line_alpha":0.1,"line_color":"#1f77b4","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"4669","type":"Line"},{"attributes":{"callback":null},"id":"4565","type":"HoverTool"},{"attributes":{"source":{"id":"4632"}},"id":"4636","type":"CDSView"},{"attributes":{"data_source":{"id":"4612"},"glyph":{"id":"4613"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4614"},"selection_glyph":null,"view":{"id":"4616"}},"id":"4615","type":"GlyphRenderer"},{"attributes":{"source":{"id":"4657"}},"id":"4661","type":"CDSView"},{"attributes":{},"id":"4717","type":"Selection"},{"attributes":{"data_source":{"id":"4627"},"glyph":{"id":"4628"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4629"},"selection_glyph":null,"view":{"id":"4631"}},"id":"4630","type":"GlyphRenderer"},{"attributes":{"data":{"x":[-2.358040118461496,10.910290467635015],"y":[1.95,1.95]},"selected":{"id":"4739"},"selection_policy":{"id":"4740"}},"id":"4652","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":3.4162688471038556},"y":{"value":2.55}},"id":"4694","type":"Circle"},{"attributes":{},"id":"4563","type":"UndoTool"},{"attributes":{},"id":"4740","type":"UnionRenderers"},{"attributes":{},"id":"4714","type":"UnionRenderers"},{"attributes":{"fill_color":{"value":"#ff7f0e"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.409527442378456},"y":{"value":0.3}},"id":"4603","type":"Circle"},{"attributes":{},"id":"4751","type":"Selection"},{"attributes":{"data_source":{"id":"4672"},"glyph":{"id":"4673"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4674"},"selection_glyph":null,"view":{"id":"4676"}},"id":"4675","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#ff7f0e"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.59253463805363},"y":{"value":0.6}},"id":"4619","type":"Circle"},{"attributes":{"line_color":"#ff7f0e","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"4613","type":"Line"},{"attributes":{},"id":"4732","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"4632"},"glyph":{"id":"4633"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4634"},"selection_glyph":null,"view":{"id":"4636"}},"id":"4635","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"4642"},"glyph":{"id":"4643"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4644"},"selection_glyph":null,"view":{"id":"4646"}},"id":"4645","type":"GlyphRenderer"},{"attributes":{},"id":"4731","type":"Selection"},{"attributes":{"line_alpha":0.1,"line_color":"#ff7f0e","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"4624","type":"Line"},{"attributes":{},"id":"4741","type":"Selection"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":3.4162688471038556},"y":{"value":2.55}},"id":"4693","type":"Circle"},{"attributes":{"line_color":"#ff7f0e","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"4608","type":"Line"},{"attributes":{"source":{"id":"4642"}},"id":"4646","type":"CDSView"},{"attributes":{},"id":"4723","type":"Selection"},{"attributes":{},"id":"4744","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"4617"},"glyph":{"id":"4618"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4619"},"selection_glyph":null,"view":{"id":"4621"}},"id":"4620","type":"GlyphRenderer"},{"attributes":{"source":{"id":"4647"}},"id":"4651","type":"CDSView"},{"attributes":{},"id":"4719","type":"Selection"},{"attributes":{},"id":"4754","type":"UnionRenderers"},{"attributes":{"source":{"id":"4692"}},"id":"4696","type":"CDSView"},{"attributes":{"line_color":"#ff7f0e","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"4623","type":"Line"},{"attributes":{},"id":"4746","type":"UnionRenderers"},{"attributes":{"line_alpha":0.1,"line_color":"#1f77b4","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"4689","type":"Line"},{"attributes":{},"id":"4755","type":"Selection"},{"attributes":{"children":[{"id":"4760"},{"id":"4758"}]},"id":"4761","type":"Column"},{"attributes":{"line_alpha":0.1,"line_color":"#1f77b4","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"4644","type":"Line"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#ff7f0e"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.409527442378456},"y":{"value":0.3}},"id":"4604","type":"Circle"},{"attributes":{"data":{"x":[-2.061896746944455,11.515205993744095],"y":[2.25,2.25]},"selected":{"id":"4745"},"selection_policy":{"id":"4746"}},"id":"4667","type":"ColumnDataSource"},{"attributes":{"data":{"x":[-1.4753067334040253,11.208804111875878],"y":[0.6,0.6]},"selected":{"id":"4721"},"selection_policy":{"id":"4722"}},"id":"4607","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"4692"},"glyph":{"id":"4693"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4694"},"selection_glyph":null,"view":{"id":"4696"}},"id":"4695","type":"GlyphRenderer"},{"attributes":{"line_alpha":0.1,"line_color":"#1f77b4","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"4639","type":"Line"},{"attributes":{"data":{"x":[2.393576367885318,6.880909766712211],"y":[1.95,1.95]},"selected":{"id":"4741"},"selection_policy":{"id":"4742"}},"id":"4657","type":"ColumnDataSource"},{"attributes":{"data":{"x":[-1.9958943490877263,9.311550558002505],"y":[1.65,1.65]},"selected":{"id":"4733"},"selection_policy":{"id":"4734"}},"id":"4637","type":"ColumnDataSource"},{"attributes":{},"id":"4752","type":"UnionRenderers"},{"attributes":{"line_alpha":0.1,"line_color":"#1f77b4","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"4659","type":"Line"},{"attributes":{"children":[[{"id":"4541"},0,0]]},"id":"4758","type":"GridBox"},{"attributes":{"formatter":{"id":"4706"},"ticker":{"id":"4551"}},"id":"4550","type":"LinearAxis"},{"attributes":{"data":{},"selected":{"id":"4713"},"selection_policy":{"id":"4714"}},"id":"4587","type":"ColumnDataSource"},{"attributes":{"overlay":{"id":"4566"}},"id":"4560","type":"BoxZoomTool"},{"attributes":{"text":"94.0% HDI"},"id":"4697","type":"Title"},{"attributes":{},"id":"4722","type":"UnionRenderers"},{"attributes":{"line_alpha":0.1,"line_color":"#1f77b4","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"4684","type":"Line"},{"attributes":{"overlay":{"id":"4567"}},"id":"4562","type":"LassoSelectTool"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.181352887007439},"y":{"value":2.25}},"id":"4679","type":"Circle"},{"attributes":{},"id":"4706","type":"BasicTickFormatter"},{"attributes":{"data":{"x":[2.2808674873161356,6.6262125124574265],"y":[0.3,0.3]},"selected":{"id":"4717"},"selection_policy":{"id":"4718"}},"id":"4597","type":"ColumnDataSource"},{"attributes":{"data":{"x":[2.454317217804662,6.859709821773684],"y":[0,0]},"selected":{"id":"4711"},"selection_policy":{"id":"4712"}},"id":"4582","type":"ColumnDataSource"},{"attributes":{"axis":{"id":"4550"},"ticker":null},"id":"4553","type":"Grid"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"4567","type":"PolyAnnotation"},{"attributes":{"fill_color":{"value":"#ff7f0e"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.663971371122512},"y":{"value":0}},"id":"4588","type":"Circle"},{"attributes":{},"id":"4546","type":"LinearScale"},{"attributes":{},"id":"4734","type":"UnionRenderers"},{"attributes":{"source":{"id":"4582"}},"id":"4586","type":"CDSView"},{"attributes":{"data_source":{"id":"4582"},"glyph":{"id":"4583"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4584"},"selection_glyph":null,"view":{"id":"4586"}},"id":"4585","type":"GlyphRenderer"},{"attributes":{},"id":"4726","type":"UnionRenderers"},{"attributes":{},"id":"4559","type":"PanTool"},{"attributes":{"data_source":{"id":"4587"},"glyph":{"id":"4588"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4589"},"selection_glyph":null,"view":{"id":"4591"}},"id":"4590","type":"GlyphRenderer"},{"attributes":{},"id":"4548","type":"LinearScale"},{"attributes":{},"id":"4721","type":"Selection"},{"attributes":{},"id":"4727","type":"Selection"},{"attributes":{},"id":"4735","type":"Selection"},{"attributes":{"source":{"id":"4682"}},"id":"4686","type":"CDSView"},{"attributes":{"line_color":"#ff7f0e","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"4583","type":"Line"},{"attributes":{"data":{"x":[-2.13476994033137,10.008689721107494],"y":[0.3,0.3]},"selected":{"id":"4715"},"selection_policy":{"id":"4716"}},"id":"4592","type":"ColumnDataSource"},{"attributes":{"source":{"id":"4677"}},"id":"4681","type":"CDSView"},{"attributes":{},"id":"4558","type":"ResetTool"},{"attributes":{"source":{"id":"4587"}},"id":"4591","type":"CDSView"},{"attributes":{},"id":"4728","type":"UnionRenderers"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"4566","type":"BoxAnnotation"},{"attributes":{"line_alpha":0.1,"line_color":"#ff7f0e","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"4584","type":"Line"},{"attributes":{"bounds":"auto","min_interval":1},"id":"4699","type":"DataRange1d"},{"attributes":{"formatter":{"id":"4704"},"major_label_overrides":{"0.44999999999999996":"Non Centered: mu","2.0999999999999996":"Centered: mu"},"ticker":{"id":"4701"}},"id":"4554","type":"LinearAxis"},{"attributes":{"line_alpha":0.1,"line_color":"#ff7f0e","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"4579","type":"Line"},{"attributes":{},"id":"4724","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"4677"},"glyph":{"id":"4678"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4679"},"selection_glyph":null,"view":{"id":"4681"}},"id":"4680","type":"GlyphRenderer"},{"attributes":{"source":{"id":"4577"}},"id":"4581","type":"CDSView"},{"attributes":{},"id":"4551","type":"BasicTicker"},{"attributes":{"axis":{"id":"4554"},"dimension":1,"grid_line_color":null,"ticker":null},"id":"4557","type":"Grid"},{"attributes":{"line_color":"#ff7f0e","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"4578","type":"Line"},{"attributes":{"data_source":{"id":"4592"},"glyph":{"id":"4593"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4594"},"selection_glyph":null,"view":{"id":"4596"}},"id":"4595","type":"GlyphRenderer"},{"attributes":{"line_alpha":0.1,"line_color":"#ff7f0e","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"4599","type":"Line"},{"attributes":{"data_source":{"id":"4607"},"glyph":{"id":"4608"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4609"},"selection_glyph":null,"view":{"id":"4611"}},"id":"4610","type":"GlyphRenderer"},{"attributes":{},"id":"4561","type":"WheelZoomTool"},{"attributes":{"data_source":{"id":"4597"},"glyph":{"id":"4598"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4599"},"selection_glyph":null,"view":{"id":"4601"}},"id":"4600","type":"GlyphRenderer"},{"attributes":{},"id":"4725","type":"Selection"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#ff7f0e"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.663971371122512},"y":{"value":0}},"id":"4589","type":"Circle"},{"attributes":{},"id":"4737","type":"Selection"},{"attributes":{"source":{"id":"4592"}},"id":"4596","type":"CDSView"},{"attributes":{},"id":"4720","type":"UnionRenderers"},{"attributes":{"line_alpha":0.1,"line_color":"#ff7f0e","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"4594","type":"Line"},{"attributes":{"line_color":"#ff7f0e","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"4598","type":"Line"},{"attributes":{"data":{"x":[-2.2641273152868076,10.475301593619335],"y":[0.8999999999999999,0.8999999999999999]},"selected":{"id":"4727"},"selection_policy":{"id":"4728"}},"id":"4622","type":"ColumnDataSource"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.181352887007439},"y":{"value":2.25}},"id":"4678","type":"Circle"},{"attributes":{"line_color":"#1f77b4","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"4683","type":"Line"},{"attributes":{},"id":"4564","type":"SaveTool"},{"attributes":{},"id":"4736","type":"UnionRenderers"},{"attributes":{"line_color":"#ff7f0e","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"4593","type":"Line"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"4558"},{"id":"4559"},{"id":"4560"},{"id":"4561"},{"id":"4562"},{"id":"4563"},{"id":"4564"},{"id":"4565"}]},"id":"4568","type":"Toolbar"},{"attributes":{"data":{"x":[-1.0798320889339172,10.200853218312133],"y":[0,0]},"selected":{"id":"4709"},"selection_policy":{"id":"4710"}},"id":"4577","type":"ColumnDataSource"},{"attributes":{"below":[{"id":"4550"}],"center":[{"id":"4553"},{"id":"4557"}],"left":[{"id":"4554"}],"outline_line_color":null,"output_backend":"webgl","plot_height":435,"plot_width":360,"renderers":[{"id":"4580"},{"id":"4585"},{"id":"4590"},{"id":"4595"},{"id":"4600"},{"id":"4605"},{"id":"4610"},{"id":"4615"},{"id":"4620"},{"id":"4625"},{"id":"4630"},{"id":"4635"},{"id":"4640"},{"id":"4645"},{"id":"4650"},{"id":"4655"},{"id":"4660"},{"id":"4665"},{"id":"4670"},{"id":"4675"},{"id":"4680"},{"id":"4685"},{"id":"4690"},{"id":"4695"}],"title":{"id":"4697"},"toolbar":{"id":"4568"},"toolbar_location":null,"x_range":{"id":"4699"},"x_scale":{"id":"4546"},"y_range":{"id":"4700"},"y_scale":{"id":"4548"}},"id":"4541","subtype":"Figure","type":"Plot"}],"root_ids":["4761"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"9ec5bc75-8df5-480c-b20a-49d19d69bf8e","root_ids":["4761"],"roots":{"4761":"068cb474-8c96-4e3e-80e7-288b090ad2d9"}}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);
                
                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    var attempts = 0;
                    var timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        },
        function(Bokeh) {
        
        
        }
      ];
    
      function run_inline_js() {
        
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i].call(root, root.Bokeh);
        }
        
      }
    
      if (root._bokeh_is_loading === 0) {
        console.debug("Bokeh: BokehJS loaded, going straight to plotting");
        run_inline_js();
      } else {
        load_libs(css_urls, js_urls, function() {
          console.debug("Bokeh: BokehJS plotting callback run at", now());
          run_inline_js();
        });
      }
    }(window));
  };
  if (document.readyState != "loading") fn();
  else document.addEventListener("DOMContentLoaded", fn);
})();