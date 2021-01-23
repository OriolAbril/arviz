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
    
      
      
    
      var element = document.getElementById("4f103eb9-0fd8-4f93-88ab-14c47bcc3ddd");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '4f103eb9-0fd8-4f93-88ab-14c47bcc3ddd' but no matching script tag was found.")
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
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.3.min.js": "T2yuo9Oe71Cz/I4X9Ac5+gpEa5a8PpJCDlqKYO0CfAuEszu1JrXLl8YugMqYe3sM", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.3.min.js": "98GDGJ0kOMCUMUePhksaQ/GYgB3+NH9h996V88sh3aOiUNX3N+fLXAtry6xctSZ6", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.3.min.js": "89bArO+nlbP3sgakeHjCo1JYxYR5wufVgA3IbUvDY+K7w4zyxJqssu7wVnfeKCq8"};
    
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
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.3.min.js"];
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
                    
                  var docs_json = '{"d6d6305f-20da-47bf-bf6f-c419b11addb3":{"roots":{"references":[{"attributes":{},"id":"4631","type":"Selection"},{"attributes":{"data_source":{"id":"4600"},"glyph":{"id":"4601"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4602"},"selection_glyph":null,"view":{"id":"4604"}},"id":"4603","type":"GlyphRenderer"},{"attributes":{},"id":"4632","type":"UnionRenderers"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":3.4162688471038556},"y":{"value":2.55}},"id":"4602","type":"Circle"},{"attributes":{"children":[[{"id":"4449"},0,0]]},"id":"4666","type":"GridBox"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"4474","type":"BoxAnnotation"},{"attributes":{"source":{"id":"4600"}},"id":"4604","type":"CDSView"},{"attributes":{},"id":"4471","type":"UndoTool"},{"attributes":{},"id":"4633","type":"Selection"},{"attributes":{},"id":"4634","type":"UnionRenderers"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"4475","type":"PolyAnnotation"},{"attributes":{"line_alpha":0.1,"line_color":"#ff7f0e","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"4487","type":"Line"},{"attributes":{"overlay":{"id":"4474"}},"id":"4468","type":"BoxZoomTool"},{"attributes":{},"id":"4466","type":"ResetTool"},{"attributes":{},"id":"4635","type":"Selection"},{"attributes":{},"id":"4472","type":"SaveTool"},{"attributes":{},"id":"4636","type":"UnionRenderers"},{"attributes":{},"id":"4467","type":"PanTool"},{"attributes":{"axis":{"id":"4462"},"dimension":1,"grid_line_color":null,"ticker":null},"id":"4465","type":"Grid"},{"attributes":{},"id":"4456","type":"LinearScale"},{"attributes":{},"id":"4459","type":"BasicTicker"},{"attributes":{"data":{"x":[-1.0798320889339172,10.200853218312133],"y":[0,0]},"selected":{"id":"4617"},"selection_policy":{"id":"4618"}},"id":"4485","type":"ColumnDataSource"},{"attributes":{},"id":"4637","type":"Selection"},{"attributes":{},"id":"4638","type":"UnionRenderers"},{"attributes":{"formatter":{"id":"4614"},"ticker":{"id":"4459"}},"id":"4458","type":"LinearAxis"},{"attributes":{"text":"94.0% HDI"},"id":"4605","type":"Title"},{"attributes":{},"id":"4639","type":"Selection"},{"attributes":{},"id":"4640","type":"UnionRenderers"},{"attributes":{"toolbars":[{"id":"4476"}],"tools":[{"id":"4466"},{"id":"4467"},{"id":"4468"},{"id":"4469"},{"id":"4470"},{"id":"4471"},{"id":"4472"},{"id":"4473"}]},"id":"4667","type":"ProxyToolbar"},{"attributes":{"line_color":"#ff7f0e","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"4486","type":"Line"},{"attributes":{"axis":{"id":"4458"},"ticker":null},"id":"4461","type":"Grid"},{"attributes":{},"id":"4641","type":"Selection"},{"attributes":{},"id":"4642","type":"UnionRenderers"},{"attributes":{"callback":null},"id":"4473","type":"HoverTool"},{"attributes":{"data_source":{"id":"4500"},"glyph":{"id":"4501"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4502"},"selection_glyph":null,"view":{"id":"4504"}},"id":"4503","type":"GlyphRenderer"},{"attributes":{},"id":"4643","type":"Selection"},{"attributes":{},"id":"4644","type":"UnionRenderers"},{"attributes":{"formatter":{"id":"4612"},"major_label_overrides":{"0.44999999999999996":"Non Centered: mu","2.0999999999999996":"Centered: mu"},"ticker":{"id":"4609"}},"id":"4462","type":"LinearAxis"},{"attributes":{"data_source":{"id":"4485"},"glyph":{"id":"4486"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4487"},"selection_glyph":null,"view":{"id":"4489"}},"id":"4488","type":"GlyphRenderer"},{"attributes":{"source":{"id":"4595"}},"id":"4599","type":"CDSView"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"4466"},{"id":"4467"},{"id":"4468"},{"id":"4469"},{"id":"4470"},{"id":"4471"},{"id":"4472"},{"id":"4473"}]},"id":"4476","type":"Toolbar"},{"attributes":{},"id":"4645","type":"Selection"},{"attributes":{},"id":"4646","type":"UnionRenderers"},{"attributes":{},"id":"4454","type":"LinearScale"},{"attributes":{"below":[{"id":"4458"}],"center":[{"id":"4461"},{"id":"4465"}],"left":[{"id":"4462"}],"outline_line_color":null,"output_backend":"webgl","plot_height":435,"plot_width":360,"renderers":[{"id":"4488"},{"id":"4493"},{"id":"4498"},{"id":"4503"},{"id":"4508"},{"id":"4513"},{"id":"4518"},{"id":"4523"},{"id":"4528"},{"id":"4533"},{"id":"4538"},{"id":"4543"},{"id":"4548"},{"id":"4553"},{"id":"4558"},{"id":"4563"},{"id":"4568"},{"id":"4573"},{"id":"4578"},{"id":"4583"},{"id":"4588"},{"id":"4593"},{"id":"4598"},{"id":"4603"}],"title":{"id":"4605"},"toolbar":{"id":"4476"},"toolbar_location":null,"x_range":{"id":"4607"},"x_scale":{"id":"4454"},"y_range":{"id":"4608"},"y_scale":{"id":"4456"}},"id":"4449","subtype":"Figure","type":"Plot"},{"attributes":{"source":{"id":"4485"}},"id":"4489","type":"CDSView"},{"attributes":{},"id":"4661","type":"Selection"},{"attributes":{"line_color":"#ff7f0e","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"4536","type":"Line"},{"attributes":{"line_color":"#ff7f0e","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"4531","type":"Line"},{"attributes":{"data_source":{"id":"4535"},"glyph":{"id":"4536"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4537"},"selection_glyph":null,"view":{"id":"4539"}},"id":"4538","type":"GlyphRenderer"},{"attributes":{"line_color":"#1f77b4","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"4546","type":"Line"},{"attributes":{},"id":"4647","type":"Selection"},{"attributes":{},"id":"4662","type":"UnionRenderers"},{"attributes":{"source":{"id":"4530"}},"id":"4534","type":"CDSView"},{"attributes":{},"id":"4648","type":"UnionRenderers"},{"attributes":{"line_alpha":0.1,"line_color":"#ff7f0e","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"4532","type":"Line"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#ff7f0e"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.3706630373472235},"y":{"value":0.8999999999999999}},"id":"4542","type":"Circle"},{"attributes":{},"id":"4617","type":"Selection"},{"attributes":{},"id":"4612","type":"BasicTickFormatter"},{"attributes":{},"id":"4618","type":"UnionRenderers"},{"attributes":{},"id":"4663","type":"Selection"},{"attributes":{"fill_color":{"value":"#ff7f0e"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.3706630373472235},"y":{"value":0.8999999999999999}},"id":"4541","type":"Circle"},{"attributes":{},"id":"4664","type":"UnionRenderers"},{"attributes":{"data":{"x":[2.2921199846409115,6.479187946875487],"y":[0.8999999999999999,0.8999999999999999]},"selected":{"id":"4637"},"selection_policy":{"id":"4638"}},"id":"4535","type":"ColumnDataSource"},{"attributes":{"line_color":"#ff7f0e","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"4516","type":"Line"},{"attributes":{"source":{"id":"4535"}},"id":"4539","type":"CDSView"},{"attributes":{},"id":"4469","type":"WheelZoomTool"},{"attributes":{"line_alpha":0.1,"line_color":"#ff7f0e","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"4537","type":"Line"},{"attributes":{"data_source":{"id":"4540"},"glyph":{"id":"4541"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4542"},"selection_glyph":null,"view":{"id":"4544"}},"id":"4543","type":"GlyphRenderer"},{"attributes":{"line_alpha":0.1,"line_color":"#ff7f0e","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"4492","type":"Line"},{"attributes":{"bounds":"auto","end":3.4499999999999997,"min_interval":2,"start":-0.8999999999999999},"id":"4608","type":"DataRange1d"},{"attributes":{"data_source":{"id":"4515"},"glyph":{"id":"4516"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4517"},"selection_glyph":null,"view":{"id":"4519"}},"id":"4518","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"4550"},"glyph":{"id":"4551"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4552"},"selection_glyph":null,"view":{"id":"4554"}},"id":"4553","type":"GlyphRenderer"},{"attributes":{"toolbar":{"id":"4667"},"toolbar_location":"above"},"id":"4668","type":"ToolbarBox"},{"attributes":{"data":{},"selected":{"id":"4639"},"selection_policy":{"id":"4640"}},"id":"4540","type":"ColumnDataSource"},{"attributes":{},"id":"4649","type":"Selection"},{"attributes":{"data":{"x":[-2.358040118461496,10.910290467635015],"y":[1.95,1.95]},"selected":{"id":"4647"},"selection_policy":{"id":"4648"}},"id":"4560","type":"ColumnDataSource"},{"attributes":{},"id":"4650","type":"UnionRenderers"},{"attributes":{"source":{"id":"4540"}},"id":"4544","type":"CDSView"},{"attributes":{},"id":"4619","type":"Selection"},{"attributes":{"fill_color":{"value":"#ff7f0e"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.663971371122512},"y":{"value":0}},"id":"4496","type":"Circle"},{"attributes":{},"id":"4620","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"4560"},"glyph":{"id":"4561"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4562"},"selection_glyph":null,"view":{"id":"4564"}},"id":"4563","type":"GlyphRenderer"},{"attributes":{"line_color":"#ff7f0e","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"4506","type":"Line"},{"attributes":{"line_color":"#1f77b4","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"4551","type":"Line"},{"attributes":{"line_color":"#1f77b4","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"4561","type":"Line"},{"attributes":{"source":{"id":"4545"}},"id":"4549","type":"CDSView"},{"attributes":{"line_alpha":0.1,"line_color":"#1f77b4","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"4547","type":"Line"},{"attributes":{"data":{"x":[-1.4753067334040253,11.208804111875878],"y":[0.6,0.6]},"selected":{"id":"4629"},"selection_policy":{"id":"4630"}},"id":"4515","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":3.374380781729923},"y":{"value":1.65}},"id":"4557","type":"Circle"},{"attributes":{},"id":"4651","type":"Selection"},{"attributes":{"line_alpha":0.1,"line_color":"#ff7f0e","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"4502","type":"Line"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":3.374380781729923},"y":{"value":1.65}},"id":"4556","type":"Circle"},{"attributes":{},"id":"4652","type":"UnionRenderers"},{"attributes":{"data":{"x":[0.9649316231388204,5.949680273009057],"y":[1.65,1.65]},"selected":{"id":"4643"},"selection_policy":{"id":"4644"}},"id":"4550","type":"ColumnDataSource"},{"attributes":{"source":{"id":"4550"}},"id":"4554","type":"CDSView"},{"attributes":{},"id":"4621","type":"Selection"},{"attributes":{"data":{"x":[-1.9958943490877263,9.311550558002505],"y":[1.65,1.65]},"selected":{"id":"4641"},"selection_policy":{"id":"4642"}},"id":"4545","type":"ColumnDataSource"},{"attributes":{"line_alpha":0.1,"line_color":"#1f77b4","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"4552","type":"Line"},{"attributes":{},"id":"4622","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"4555"},"glyph":{"id":"4556"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4557"},"selection_glyph":null,"view":{"id":"4559"}},"id":"4558","type":"GlyphRenderer"},{"attributes":{"data":{},"selected":{"id":"4621"},"selection_policy":{"id":"4622"}},"id":"4495","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"4490"},"glyph":{"id":"4491"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4492"},"selection_glyph":null,"view":{"id":"4494"}},"id":"4493","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"4565"},"glyph":{"id":"4566"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4567"},"selection_glyph":null,"view":{"id":"4569"}},"id":"4568","type":"GlyphRenderer"},{"attributes":{"data":{},"selected":{"id":"4645"},"selection_policy":{"id":"4646"}},"id":"4555","type":"ColumnDataSource"},{"attributes":{"children":[{"id":"4668"},{"id":"4666"}]},"id":"4669","type":"Column"},{"attributes":{"data":{"x":[-2.061896746944455,11.515205993744095],"y":[2.25,2.25]},"selected":{"id":"4653"},"selection_policy":{"id":"4654"}},"id":"4575","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#ff7f0e"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.663971371122512},"y":{"value":0}},"id":"4497","type":"Circle"},{"attributes":{"source":{"id":"4555"}},"id":"4559","type":"CDSView"},{"attributes":{},"id":"4653","type":"Selection"},{"attributes":{"overlay":{"id":"4475"}},"id":"4470","type":"LassoSelectTool"},{"attributes":{"data_source":{"id":"4575"},"glyph":{"id":"4576"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4577"},"selection_glyph":null,"view":{"id":"4579"}},"id":"4578","type":"GlyphRenderer"},{"attributes":{},"id":"4654","type":"UnionRenderers"},{"attributes":{"source":{"id":"4495"}},"id":"4499","type":"CDSView"},{"attributes":{"line_color":"#1f77b4","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"4566","type":"Line"},{"attributes":{"source":{"id":"4500"}},"id":"4504","type":"CDSView"},{"attributes":{"line_color":"#1f77b4","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"4576","type":"Line"},{"attributes":{},"id":"4623","type":"Selection"},{"attributes":{"source":{"id":"4560"}},"id":"4564","type":"CDSView"},{"attributes":{},"id":"4624","type":"UnionRenderers"},{"attributes":{"line_alpha":0.1,"line_color":"#1f77b4","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"4562","type":"Line"},{"attributes":{"line_color":"#ff7f0e","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"4501","type":"Line"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.74129666559688},"y":{"value":1.95}},"id":"4572","type":"Circle"},{"attributes":{"data":{"x":[2.454317217804662,6.859709821773684],"y":[0,0]},"selected":{"id":"4619"},"selection_policy":{"id":"4620"}},"id":"4490","type":"ColumnDataSource"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.74129666559688},"y":{"value":1.95}},"id":"4571","type":"Circle"},{"attributes":{"source":{"id":"4490"}},"id":"4494","type":"CDSView"},{"attributes":{"data":{"x":[2.393576367885318,6.880909766712211],"y":[1.95,1.95]},"selected":{"id":"4649"},"selection_policy":{"id":"4650"}},"id":"4565","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#ff7f0e"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.409527442378456},"y":{"value":0.3}},"id":"4512","type":"Circle"},{"attributes":{"data":{"x":[-2.2641273152868076,10.475301593619335],"y":[0.8999999999999999,0.8999999999999999]},"selected":{"id":"4635"},"selection_policy":{"id":"4636"}},"id":"4530","type":"ColumnDataSource"},{"attributes":{"source":{"id":"4565"}},"id":"4569","type":"CDSView"},{"attributes":{"line_alpha":0.1,"line_color":"#1f77b4","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"4567","type":"Line"},{"attributes":{},"id":"4655","type":"Selection"},{"attributes":{"data_source":{"id":"4505"},"glyph":{"id":"4506"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4507"},"selection_glyph":null,"view":{"id":"4509"}},"id":"4508","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"4570"},"glyph":{"id":"4571"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4572"},"selection_glyph":null,"view":{"id":"4574"}},"id":"4573","type":"GlyphRenderer"},{"attributes":{},"id":"4656","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"4580"},"glyph":{"id":"4581"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4582"},"selection_glyph":null,"view":{"id":"4584"}},"id":"4583","type":"GlyphRenderer"},{"attributes":{},"id":"4625","type":"Selection"},{"attributes":{"line_color":"#ff7f0e","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"4491","type":"Line"},{"attributes":{"data":{},"selected":{"id":"4651"},"selection_policy":{"id":"4652"}},"id":"4570","type":"ColumnDataSource"},{"attributes":{},"id":"4626","type":"UnionRenderers"},{"attributes":{"data":{"x":[-0.7842313478998125,9.985046696913068],"y":[2.55,2.55]},"selected":{"id":"4659"},"selection_policy":{"id":"4660"}},"id":"4590","type":"ColumnDataSource"},{"attributes":{"source":{"id":"4570"}},"id":"4574","type":"CDSView"},{"attributes":{"data_source":{"id":"4495"},"glyph":{"id":"4496"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4497"},"selection_glyph":null,"view":{"id":"4499"}},"id":"4498","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"4590"},"glyph":{"id":"4591"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4592"},"selection_glyph":null,"view":{"id":"4594"}},"id":"4593","type":"GlyphRenderer"},{"attributes":{"line_color":"#1f77b4","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"4581","type":"Line"},{"attributes":{"fill_color":{"value":"#ff7f0e"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.409527442378456},"y":{"value":0.3}},"id":"4511","type":"Circle"},{"attributes":{"line_color":"#1f77b4","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"4591","type":"Line"},{"attributes":{"data":{"x":[2.2808674873161356,6.6262125124574265],"y":[0.3,0.3]},"selected":{"id":"4625"},"selection_policy":{"id":"4626"}},"id":"4505","type":"ColumnDataSource"},{"attributes":{"source":{"id":"4575"}},"id":"4579","type":"CDSView"},{"attributes":{},"id":"4657","type":"Selection"},{"attributes":{"source":{"id":"4505"}},"id":"4509","type":"CDSView"},{"attributes":{"line_alpha":0.1,"line_color":"#1f77b4","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"4577","type":"Line"},{"attributes":{},"id":"4658","type":"UnionRenderers"},{"attributes":{"line_alpha":0.1,"line_color":"#ff7f0e","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"4507","type":"Line"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.181352887007439},"y":{"value":2.25}},"id":"4587","type":"Circle"},{"attributes":{"data_source":{"id":"4510"},"glyph":{"id":"4511"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4512"},"selection_glyph":null,"view":{"id":"4514"}},"id":"4513","type":"GlyphRenderer"},{"attributes":{},"id":"4627","type":"Selection"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.181352887007439},"y":{"value":2.25}},"id":"4586","type":"Circle"},{"attributes":{},"id":"4628","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"4530"},"glyph":{"id":"4531"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4532"},"selection_glyph":null,"view":{"id":"4534"}},"id":"4533","type":"GlyphRenderer"},{"attributes":{"data":{"x":[1.9127415870254314,7.0411499662839026],"y":[2.25,2.25]},"selected":{"id":"4655"},"selection_policy":{"id":"4656"}},"id":"4580","type":"ColumnDataSource"},{"attributes":{"data":{},"selected":{"id":"4627"},"selection_policy":{"id":"4628"}},"id":"4510","type":"ColumnDataSource"},{"attributes":{"source":{"id":"4580"}},"id":"4584","type":"CDSView"},{"attributes":{"line_alpha":0.1,"line_color":"#1f77b4","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"4582","type":"Line"},{"attributes":{"data_source":{"id":"4585"},"glyph":{"id":"4586"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4587"},"selection_glyph":null,"view":{"id":"4589"}},"id":"4588","type":"GlyphRenderer"},{"attributes":{"source":{"id":"4510"}},"id":"4514","type":"CDSView"},{"attributes":{"source":{"id":"4525"}},"id":"4529","type":"CDSView"},{"attributes":{"data_source":{"id":"4595"},"glyph":{"id":"4596"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4597"},"selection_glyph":null,"view":{"id":"4599"}},"id":"4598","type":"GlyphRenderer"},{"attributes":{"fill_color":{"value":"#ff7f0e"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.59253463805363},"y":{"value":0.6}},"id":"4526","type":"Circle"},{"attributes":{"data":{},"selected":{"id":"4633"},"selection_policy":{"id":"4634"}},"id":"4525","type":"ColumnDataSource"},{"attributes":{"data":{},"selected":{"id":"4657"},"selection_policy":{"id":"4658"}},"id":"4585","type":"ColumnDataSource"},{"attributes":{},"id":"4659","type":"Selection"},{"attributes":{"bounds":"auto","min_interval":1},"id":"4607","type":"DataRange1d"},{"attributes":{"data_source":{"id":"4520"},"glyph":{"id":"4521"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4522"},"selection_glyph":null,"view":{"id":"4524"}},"id":"4523","type":"GlyphRenderer"},{"attributes":{},"id":"4660","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"4545"},"glyph":{"id":"4546"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4547"},"selection_glyph":null,"view":{"id":"4549"}},"id":"4548","type":"GlyphRenderer"},{"attributes":{"source":{"id":"4585"}},"id":"4589","type":"CDSView"},{"attributes":{"source":{"id":"4515"}},"id":"4519","type":"CDSView"},{"attributes":{},"id":"4629","type":"Selection"},{"attributes":{"ticks":[0.44999999999999996,2.0999999999999996]},"id":"4609","type":"FixedTicker"},{"attributes":{"line_alpha":0.1,"line_color":"#ff7f0e","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"4517","type":"Line"},{"attributes":{},"id":"4630","type":"UnionRenderers"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#ff7f0e"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.59253463805363},"y":{"value":0.6}},"id":"4527","type":"Circle"},{"attributes":{"line_color":"#1f77b4","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"4596","type":"Line"},{"attributes":{"data":{"x":[2.1521098502407368,6.9643589964055215],"y":[0.6,0.6]},"selected":{"id":"4631"},"selection_policy":{"id":"4632"}},"id":"4520","type":"ColumnDataSource"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":3.4162688471038556},"y":{"value":2.55}},"id":"4601","type":"Circle"},{"attributes":{"data":{"x":[-2.13476994033137,10.008689721107494],"y":[0.3,0.3]},"selected":{"id":"4623"},"selection_policy":{"id":"4624"}},"id":"4500","type":"ColumnDataSource"},{"attributes":{"source":{"id":"4590"}},"id":"4594","type":"CDSView"},{"attributes":{"line_color":"#ff7f0e","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"4521","type":"Line"},{"attributes":{"line_alpha":0.1,"line_color":"#1f77b4","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"4592","type":"Line"},{"attributes":{},"id":"4614","type":"BasicTickFormatter"},{"attributes":{"source":{"id":"4520"}},"id":"4524","type":"CDSView"},{"attributes":{"line_alpha":0.1,"line_color":"#1f77b4","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"4597","type":"Line"},{"attributes":{"line_alpha":0.1,"line_color":"#ff7f0e","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"4522","type":"Line"},{"attributes":{"data":{},"selected":{"id":"4663"},"selection_policy":{"id":"4664"}},"id":"4600","type":"ColumnDataSource"},{"attributes":{"data":{"x":[1.9790083397469873,5.455951625836456],"y":[2.55,2.55]},"selected":{"id":"4661"},"selection_policy":{"id":"4662"}},"id":"4595","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"4525"},"glyph":{"id":"4526"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4527"},"selection_glyph":null,"view":{"id":"4529"}},"id":"4528","type":"GlyphRenderer"}],"root_ids":["4669"]},"title":"Bokeh Application","version":"2.2.3"}}';
                  var render_items = [{"docid":"d6d6305f-20da-47bf-bf6f-c419b11addb3","root_ids":["4669"],"roots":{"4669":"4f103eb9-0fd8-4f93-88ab-14c47bcc3ddd"}}];
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